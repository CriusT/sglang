"""
Mock implementation of the libfabric_client module for debugging purposes.
This file mirrors the API defined in pybind_client.cpp but with empty function implementations.

This mock simulates the RDMA-based client-server communication system for tensor operations.
It provides a Python-friendly interface to what would normally be C++ RDMA functionality.
"""
import time
import torch
import logging
from typing import List, Optional, Tuple
from sglang.srt.eaas.eaas_server_manager import EaasServerManager

logger = logging.getLogger(__name__)
class EaasMockClient:
    """
    Mock implementation of the PyClient class from pybind_client.cpp
    
    This class simulates a client that communicates with remote RDMA servers
    for distributed tensor operations. In the real implementation, this would use
    libfabric for RDMA communication with GPU memory on remote servers.
    """

    def __init__(self):
        self.is_connected = False
        self.num_servers = 1
        self.seed_server_last_received_tensors_dict = {} # Dict[seed, List[torch.Tensor]]
        self.client_id = -1

    def connect_to_tensor_servers_from_json(
            self, 
            json_file_path: str, 
            device_name: str = "", 
            client_id: int = 1, 
            cuda_device: int = -1) -> bool:
        self.is_connected = True
        self.client_id = client_id
        return True
    
    def mock_map_expert_to_server(self, layer: int, expert_ids: List[int]) -> List[int]:
        """
        Mock function to map experts to servers.
        """
        server_ids = []
        for i in range(len(expert_ids)):
            server_ids.append(i % self.num_servers)
        return server_ids

    def moe_request_to_servers(
        self, 
        server_indices: List[int] = None, # always size==1 for now
        seed: int = None,
        tensor: Optional[torch.Tensor] = None,
        tensor_dims: Optional[Tuple[int, int, int]] = None,
        tensor_dtype: str = "bf16",
        layer: int = 0,
        active_experts: List[int] = None
    ) -> bool:
        logger.info(f"Mock moe_request_to_servers: {server_indices}, {seed}, {tensor.shape} {layer}, {active_experts}")

        if seed not in self.seed_server_last_received_tensors_dict:
            self.seed_server_last_received_tensors_dict[seed] = [None] * self.num_servers
        self.seed_server_last_received_tensors_dict[seed][server_indices[0]] = tensor
        logger.info(f"keys after moe_request_to_servers: {list(self.seed_server_last_received_tensors_dict.keys())}")
        return True

    def wait_for_tensor_result(
        self, 
        server_indices: List[int] = None, 
        timeout_ms: int = 50000, 
        seed: int = 0
    ) -> Optional[List[torch.Tensor]]:
        logger.info(f"Mock wait_for_tensor_result: {server_indices},  {seed}")

        if seed not in self.seed_server_last_received_tensors_dict:
            logger.info("seed not in self.seed_server_last_received_tensors_dict")
            return None
        logger.info(f"keys after wait_for_tensor_result: {self.seed_server_last_received_tensors_dict.keys()}")
        return self.seed_server_last_received_tensors_dict[seed]
    


    # def get_server_addresses(self, expert_ids: List[int]) -> List[str]:
    #     return self.server_manager.choose_server_addresses(expert_ids)