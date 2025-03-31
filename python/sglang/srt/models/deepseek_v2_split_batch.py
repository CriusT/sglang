# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from:
# https://github.com/vllm-project/vllm/blob/fb6af8bc086328ca6659e72d11ffd4309ce4de22/vllm/model_executor/models/deepseek_v2.py
"""Inference-only DeepseekV2 model."""

import os
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn
import logging
from transformers import PretrainedConfig
from vllm import _custom_ops as ops

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ReplicatedLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import EPMoE
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_to_tensor_quant,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.layers.quantization.int8_utils import (
    block_dequant as int8_block_dequant,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import is_cuda_available, is_hip

from typing import List

from sglang.srt.models.deepseek_v2 import (
    DeepseekV2MLP, 
    DeepseekV2Attention, 
    DeepseekV2AttentionMLA,
    DeepseekV2MoE,
    all_gather
)

from sglang.srt.eaas.eaas_mock_client import EaasMockClient

is_hip_ = is_hip()

if is_cuda_available():
    from sgl_kernel import bmm_fp8

logger = logging.getLogger(__name__)


class DeepseekV2SplitBatchMoE(DeepseekV2MoE):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__(
            config=config,
            quant_config=quant_config,
        )
        self.top_k = config.num_experts_per_tok
        self.renormalize = config.norm_topk_prob
        self.topk_group = config.topk_group
        self.num_expert_group = config.n_group
        self.correction_bias = self.gate.e_score_correction_bias

    def forward_gate(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        # hidden_states: [num_tokens, hidden_size], torch.bfloat16
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)

        from sglang.srt.layers.moe.topk import select_experts
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            use_grouped_topk=True,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            correction_bias=self.correction_bias
        )

        return topk_weights, topk_ids
    
    def forward_experts(
        self,
        hidden_states: torch.Tensor,
        eaas_client: EaasMockClient,
        layer_id: int,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:        
        results = []
        for i in range(hidden_states.shape[0]):
            row_topk_ids = topk_ids[i:i+1]
            server_addresses = eaas_client.get_server_addresses(row_topk_ids)
            row_hidden_states = hidden_states[i:i+1]
            topk_ids_list = row_topk_ids.tolist()
            for i, server_address in enumerate(server_addresses):
                eaas_client.moe_request_with_tensor(
                    server_address=server_address,
                    hidden_states=row_hidden_states,
                    seed=0,
                    layer_id=layer_id,
                    expert_ids=topk_ids_list[i],
                )
            row_result = eaas_client.get_tensor_result()
            results.append(row_result)

        return torch.cat(results, dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(hidden_states)


class DeepseekV2SplitBatchDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.enable_dp_attention = (
            not global_server_args_dict["disable_mla"]
            and global_server_args_dict["enable_dp_attention"]
        )
        if self.enable_dp_attention:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_group = get_tp_group()
        if not global_server_args_dict["disable_mla"]:
            self.self_attn = DeepseekV2AttentionMLA(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=(
                    config.q_lora_rank if hasattr(config, "q_lora_rank") else None
                ),
                kv_lora_rank=config.kv_lora_rank,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                layer_id=layer_id,
                use_dp=self.enable_dp_attention,
            )
        else:
            self.self_attn = DeepseekV2Attention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=(
                    config.q_lora_rank if hasattr(config, "q_lora_rank") else None
                ),
                kv_lora_rank=config.kv_lora_rank,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                layer_id=layer_id,
            )
        if is_nextn or (
            config.n_routed_experts is not None
            and layer_id >= config.first_k_dense_replace
            and layer_id % config.moe_layer_freq == 0
        ):
            self.mlp = DeepseekV2SplitBatchMoE(config=config, quant_config=quant_config)
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )

        # Fully Connected
        if self.enable_dp_attention:
            hidden_states, start_idx, end_idx = all_gather(
                hidden_states, forward_batch, self.tp_rank, self.tp_size, self.tp_group
            )
            hidden_states = self.mlp(hidden_states)
            hidden_states = hidden_states[start_idx:end_idx]
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
    
    def forward_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        
        start_idx = None
        end_idx = None
        if self.enable_dp_attention:
            hidden_states, start_idx, end_idx = all_gather(
                hidden_states, forward_batch, self.tp_rank, self.tp_size, self.tp_group
            )

        return hidden_states, residual, start_idx, end_idx

    def forward_gate(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        
        topk_weights, topk_ids = self.mlp.forward_gate(hidden_states)
            
        return topk_weights, topk_ids

    def forward_experts(
        self, 
        hidden_states: torch.Tensor, 
        forward_batch: ForwardBatch, 
        residual: torch.Tensor,
        eaas_client: EaasMockClient,
        layer_id: int,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        
        hidden_states = self.mlp.forward_experts(hidden_states, eaas_client, layer_id, topk_ids)

        return hidden_states, residual


class DeepseekV2SplitBatchModel(nn.Module):

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.first_k_dense_replace = config.first_k_dense_replace

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not global_server_args_dict["enable_dp_attention"],
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV2SplitBatchDecoderLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def non_split_batch_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # same to DeepseekV2Model.forward
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )
        if not forward_batch.forward_mode.is_idle():
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        eaas_client: Optional[EaasMockClient] = None,
        stream_a: Optional[torch.cuda.Stream] = None,
        stream_b: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:

        if forward_batch.batch_size == 1 \
            or not forward_batch.forward_mode.is_decode() \
            or eaas_client is None:

            return self.non_split_batch_forward(
                input_ids, positions, forward_batch
            )
        
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        # Forward single batch for the first k dense replace layers
        # logger.info(f"Forward single batch for the first {self.first_k_dense_replace} layers")
        hidden_states, residual = self.forward_dense_layers(
            hidden_states, residual, positions, forward_batch
        )

        # Split the batch into two sub-batches
        # logger.info(f"Split the batch into two sub-batches")
        hidden_states_list, residual_list, positions_list, forward_batches_list = \
            self.split_batches(
                hidden_states, residual, positions, forward_batch, 
                split_index=forward_batch.batch_size // 2
            )

        # Forward multiple batches for the remaining layers
        # logger.info(f"Forward multiple batches for the remaining layers")
        hidden_states_list, residual_list = self.forward_moe_layers(
            hidden_states_list, residual_list, positions_list, forward_batches_list, 
            eaas_client, stream_a, stream_b
        )

        # Merge the outputs
        # logger.info(f"Merge the outputs")
        hidden_states = torch.concat(hidden_states_list, dim=0)
        residual = torch.concat(residual_list, dim=0)

        # Post-process the outputs
        # logger.info(f"Post-process the outputs")
        if not forward_batch.forward_mode.is_idle():
            hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states
        
    def forward_dense_layers(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        for i in range(self.first_k_dense_replace):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )
        return hidden_states, residual
    
    def forward_moe_layers(
        self,
        hidden_states_list: List[torch.Tensor],
        residual_list: List[torch.Tensor],
        positions_list: List[torch.Tensor],
        forward_batches: List[ForwardBatch],
        eaas_client: Optional[EaasMockClient] = None,
        stream_a: Optional[torch.cuda.Stream] = None,
        stream_b: Optional[torch.cuda.Stream] = None,
    ) -> List[torch.Tensor]:
        """
        Forward pass for multiple batches with operations overlapping across layers.
        
        Args:
            forward_batches: List of ForwardBatch objects
        
        Returns:
            List of output tensors for each batch
        """
        assert len(forward_batches) == 2, "This implementation supports exactly 2 batches"
        
        remaining_layers = len(self.layers) - self.first_k_dense_replace
        first_moe_layer = self.first_k_dense_replace

        start_idx_list = [None] * 2
        end_idx_list = [None] * 2
        topk_weights_list = [None] * 2
        topk_ids_list = [None] * 2
        
        # Step 1: Initial attention for batch 0, layer 0
        with torch.cuda.stream(stream_a):
            hidden_states_list[0], residual_list[0], start_idx_list[0], end_idx_list[0] = \
                self.layers[first_moe_layer].forward_attention(
                    positions=positions_list[0],
                    hidden_states=hidden_states_list[0],
                    forward_batch=forward_batches[0],
                    residual=residual_list[0]
                )
            topk_weights_list[0], topk_ids_list[0] = self.layers[first_moe_layer].forward_gate(hidden_states_list[0])

        torch.cuda.synchronize()
        
        # Steps 2 through 2*num_layers: Overlapped execution
        for i in range(remaining_layers * 2 - 1): #TODO: check if this is correct
            # Calculate which operations to run in this step
            if i % 2 == 0:  # Even steps (MoE batch 0 + Attention batch 1)
                b0_op, b0_layer_offset = "moe", i // 2
                b1_op, b1_layer_offset = "attn", i // 2
            else:  # Odd steps (Attention batch 0 + MoE batch 1)
                b0_op, b0_layer_offset = "attn", i // 2 + 1
                b1_op, b1_layer_offset = "moe", i // 2
            
            # Skip operations beyond the layer range
            b0_layer = first_moe_layer + b0_layer_offset
            b1_layer = first_moe_layer + b1_layer_offset
            
            # Launch valid operations in separate streams
            with torch.cuda.stream(stream_a):
                if b0_op == "attn":
                    hidden_states_list[0], residual_list[0], start_idx_list[0], end_idx_list[0] = \
                        self.layers[b0_layer].forward_attention(
                            positions=positions_list[0],
                            hidden_states=hidden_states_list[0],
                            forward_batch=forward_batches[0],
                            residual=residual_list[0]
                        )
                    topk_weights_list[0], topk_ids_list[0] = self.layers[b0_layer].forward_gate(hidden_states_list[0])
                else:  # "moe"
                    hidden_states_list[0], residual_list[0] = self.layers[b0_layer].forward_experts(
                        hidden_states=hidden_states_list[0],
                        forward_batch=forward_batches[0],
                        residual=residual_list[0],
                        eaas_client=eaas_client,
                        layer_id=b0_layer,
                        topk_ids=topk_ids_list[0]
                    )
                    # dp attention
                    hidden_states_list[0] = hidden_states_list[0][start_idx_list[0]:end_idx_list[0]] 
            with torch.cuda.stream(stream_b):
                if b1_op == "attn":
                    hidden_states_list[1], residual_list[1], start_idx_list[1], end_idx_list[1] = \
                        self.layers[b1_layer].forward_attention(
                            positions=positions_list[1],
                            hidden_states=hidden_states_list[1],
                            forward_batch=forward_batches[1],
                            residual=residual_list[1]
                        )
                    topk_weights_list[1], topk_ids_list[1] = self.layers[b1_layer].forward_gate(hidden_states_list[1])
                else:  # "moe"
                    hidden_states_list[1], residual_list[1] = self.layers[b1_layer].forward_experts(
                        hidden_states=hidden_states_list[1],
                        forward_batch=forward_batches[1],
                        residual=residual_list[1],
                        eaas_client=eaas_client,
                        layer_id=b1_layer,
                        topk_ids=topk_ids_list[1]
                    )
                    # dp attention
                    hidden_states_list[1] = hidden_states_list[1][start_idx_list[1]:end_idx_list[1]] 
            
            # Synchronize before next step to ensure correct sequencing
            torch.cuda.synchronize()
        
        # Final MoE operation for batch 1, last layer
        with torch.cuda.stream(stream_b):
            hidden_states_list[1], residual_list[1] = self.layers[-1].forward_experts(
                hidden_states=hidden_states_list[1],
                forward_batch=forward_batches[1],
                residual=residual_list[1],
                eaas_client=eaas_client,
                layer_id=len(self.layers) - 1,
                topk_ids=topk_ids_list[1]
            )
        torch.cuda.synchronize()
        
        return hidden_states_list, residual_list

    @staticmethod
    def split_batches(
        hidden_states: torch.Tensor, 
        residual: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch, 
        split_index: int # We only support 2 batches for now
    ) -> Optional[Tuple[Dict, Dict]]:
        hidden_states_0 = hidden_states[:split_index]
        hidden_states_1 = hidden_states[split_index:]

        residual_0 = residual[:split_index]
        residual_1 = residual[split_index:]

        positions_0 = positions[:split_index]
        positions_1 = positions[split_index:]

        return [hidden_states_0, hidden_states_1], \
                [residual_0, residual_1], \
                [positions_0, positions_1], \
                [forward_batch.sub_batch_0, forward_batch.sub_batch_1]


class DeepseekV2SplitBatchForCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = DeepseekV2SplitBatchModel(config, quant_config)
        if global_server_args_dict["enable_dp_attention"]:
            self.lm_head = ReplicatedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )
            self.logits_processor = LogitsProcessor(config, skip_all_gather=True)
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size, quant_config=quant_config
            )
            self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        eaas_client: Optional[EaasMockClient] = None,
        stream_a: Optional[torch.cuda.Stream] = None,
        stream_b: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, eaas_client, stream_a, stream_b)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        MoEImpl = EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE
        expert_params_mapping = MoEImpl.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # TODO(HandH1998): Modify it when nextn is supported.
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                if num_nextn_layers > 0 and name.startswith("model.layers"):
                    name_list = name.split(".")
                    if (
                        len(name_list) >= 3
                        and int(name_list[2]) >= self.config.num_hidden_layers
                    ):
                        continue
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

        if not global_server_args_dict["disable_mla"]:
            for layer_id in range(self.config.num_hidden_layers):
                self_attn = self.model.layers[layer_id].self_attn
                if hasattr(self_attn.kv_b_proj, "qweight"):
                    # AWQ compatible
                    w = ops.awq_dequantize(
                        self_attn.kv_b_proj.qweight,
                        self_attn.kv_b_proj.scales,
                        self_attn.kv_b_proj.qzeros,
                        0,
                        0,
                        0,
                    ).T
                else:
                    w = self_attn.kv_b_proj.weight
                # NOTE(HandH1998): Since `bmm_fp8` only supports per-tensor scale, we have to requantize `self_attn.kv_b_proj`.
                # This may affect the accuracy of fp8 model.
                if hasattr(self.quant_config, "weight_block_size") and w.dtype in (
                    torch.float8_e4m3fn,
                    torch.float8_e4m3fnuz,
                ):
                    weight_block_size = self.quant_config.weight_block_size
                    if weight_block_size is not None:
                        assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                        if is_hip_:
                            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                                weight=w,
                                weight_scale=self_attn.kv_b_proj.weight_scale_inv,
                                input_scale=None,
                            )
                        else:
                            weight = w
                            weight_scale = self_attn.kv_b_proj.weight_scale_inv

                        w, scale = block_quant_to_tensor_quant(
                            weight, weight_scale, weight_block_size
                        )
                        self_attn.w_scale = scale
                if (
                    hasattr(self.quant_config, "weight_block_size")
                    and w.dtype == torch.int8
                ):
                    weight_block_size = self.quant_config.weight_block_size
                    if weight_block_size is not None:
                        assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                        weight = w
                        weight_scale = self_attn.kv_b_proj.weight_scale_inv
                        w = int8_block_dequant(
                            weight, weight_scale, weight_block_size
                        ).to(torch.bfloat16)
                w_kc, w_vc = w.unflatten(
                    0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
                ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
                self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
                self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
                if (
                    hasattr(self_attn.kv_b_proj, "weight_scale")
                    and self_attn.w_scale is None
                ):
                    self_attn.w_scale = self_attn.kv_b_proj.weight_scale
                    if is_hip_:
                        self_attn.w_scale *= 2.0

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


EntryClass = [DeepseekV2SplitBatchForCausalLM]
