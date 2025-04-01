import os
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn
import logging
from transformers import PretrainedConfig
from typing import List
import sys

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


from sglang.srt.models.deepseek_v2 import (
    DeepseekV2MLP, 
    DeepseekV2Attention, 
    DeepseekV2AttentionMLA,
    DeepseekV2MoE,
    all_gather
)

from sglang.srt.eaas.eaas_mock_client import EaasMockClient


logger = logging.getLogger(__name__)

class DeepseekV2EaasMoE(DeepseekV2MoE):

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
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:        
        
        # Dict[server_address, List[row_ids]]
        server_address_row_ids_dict = {}

        # Dict[server_address, List[expert_ids]]
        server_address_expert_ids_dict = {}

        for i in range(hidden_states.shape[0]):
            row_topk_ids = topk_ids[i:i+1]
            topk_ids_list = row_topk_ids.tolist()[0]

            server_addresses = eaas_client.mock_map_expert_to_server(layer_id, topk_ids_list)

            for j, server_address in enumerate(server_addresses):
                if server_address not in server_address_row_ids_dict:
                    server_address_row_ids_dict[server_address] = []
                if server_address not in server_address_expert_ids_dict:
                    server_address_expert_ids_dict[server_address] = []
                server_address_row_ids_dict[server_address].append(i)
                server_address_expert_ids_dict[server_address].append(topk_ids_list[j])
        
        # For a expert id, we replicate one request. 
        # Therefore, the number of requests is the same as the number of experts.
        # That is to say, request_tensor.shape[0] == expert_ids
        # Each server, corresponds to multiple pairs of (request_tensor_row, expert_ids)
        # The number of pairs is the same as the number of rows in request_tensor.
        for server_address in list(server_address_row_ids_dict.keys()):
            request_tensor = hidden_states[server_address_row_ids_dict[server_address]]
            request_tensor = request_tensor.reshape(request_tensor.shape[0], 1, request_tensor.shape[1])
            expert_ids = server_address_expert_ids_dict[server_address]
            # debug
            # logger.info(f"type(request_tensor): {type(request_tensor)}, type(server_address): {type(server_address)}, type(expert_ids): {type(expert_ids)}") 
            
            success = eaas_client.moe_request_to_servers(
                server_indices=[server_address],
                tensor=request_tensor.to(torch.float16),  # TODO(boyu): need server side modification
                seed=0,
                layer=layer_id,
                active_experts=expert_ids,
            )
            if not success:
                logger.error(f"Error in sending request to server {server_address}")
                sys.exit()

        # Get results from all servers
        # Here, I assume no merging happens on the server side
        # Therefore, for each server, the number of results is the same as the number of rows in request_tensor.

        server_results = eaas_client.wait_for_tensor_result(server_address_row_ids_dict.keys())
        final_results = self.merge_results(num_rows=hidden_states.shape[0], 
                                          hidden_size=hidden_states.shape[1],
                                          server_results=server_results, 
                                          server_address_row_ids_dict=server_address_row_ids_dict)
        
        return final_results
        # return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        eaas_client: Optional[EaasMockClient] = None,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        if not global_server_args_dict["debug_activate_eaas"]:
            return super().forward(hidden_states)
        if eaas_client is None: # non-decode mode
            return super().forward(hidden_states)
        topk_weights, topk_ids = self.forward_gate(hidden_states)
        return self.forward_experts(hidden_states, eaas_client, layer_id, topk_ids, topk_weights)
    
    def merge_results(
        self, 
        num_rows: int,
        hidden_size: int,
        server_results: List[torch.Tensor], 
        server_address_row_ids_dict: Dict[str, List[int]], 
    ) -> torch.Tensor:
        # Each result corresponds to results from one server
        # assert len(server_results) == len(server_address_row_ids_dict)

        row_results = torch.zeros(num_rows, hidden_size, device=server_results[0].device, dtype=server_results[0].dtype)
        for i, server_address in enumerate(server_address_row_ids_dict.keys()):
            row_ids = server_address_row_ids_dict[server_address]
            cur_result = server_results[i]

            for j, row_id in enumerate(row_ids):
                row_results[row_id] += cur_result[j]

        return row_results        
        

class DeepseekV2EaasSingleBatchDecoderLayer(nn.Module):
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
            self.mlp = DeepseekV2EaasMoE(config=config, quant_config=quant_config)
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
        eaas_client: Optional[EaasMockClient] = None,
        layer_id: Optional[int] = None,
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
            if isinstance(self.mlp, DeepseekV2EaasMoE) and \
                forward_batch.forward_mode.is_decode():
                hidden_states = self.mlp(hidden_states, eaas_client, layer_id)
            else:
                hidden_states = self.mlp(hidden_states)
            hidden_states = hidden_states[start_idx:end_idx]
        else:
            if isinstance(self.mlp, DeepseekV2EaasMoE) and \
                forward_batch.forward_mode.is_decode():
                hidden_states = self.mlp(hidden_states, eaas_client, layer_id)
            else:
                hidden_states = self.mlp(hidden_states)

        if forward_batch.forward_mode.is_decode():
            if layer_id == 3:
                self._save_results(hidden_states)
                sys.exit()
        return hidden_states, residual

    def _save_results(self, hidden_states):
        save_path = "/gpfs/users/tianboyu/cpu001/EaaS/log/layer_3_results_eaas.pt"
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Save tensor as numpy array
            logger.info(f"Saving model results to {save_path}")
            torch.save(hidden_states, save_path)
        except Exception as e:
            logger.error(f"Failed to save results to {save_path}: {e}")


class DeepseekV2EaasSingleBatchModel(nn.Module):

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not global_server_args_dict["enable_dp_attention"],
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV2EaasSingleBatchDecoderLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        eaas_client: Optional[EaasMockClient] = None,
    ) -> torch.Tensor:

        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual, 
                eaas_client=eaas_client, layer_id=i
            )
        if not forward_batch.forward_mode.is_idle():
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

