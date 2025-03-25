from typing import List, Optional, Dict

class EaasServerManager:
    def __init__(self):
        # List of all server addresses
        self.server_addresses = [] # List[str]
        # Mapping from expert_id to list of server addresses
        self.expert_to_servers = {} # Dict[int, List[str]]

    
    def register_all(self, servers_with_experts: Dict[str, List[int]]):
        """
        Register multiple servers with their supported expert IDs.
        
        Args:
            servers_with_experts: A dictionary mapping server addresses to lists of supported expert IDs
                                {server_address: [expert_id1, expert_id2, ...], ...}
        """
        for server_address, expert_ids in servers_with_experts.items():
            self.register(server_address, expert_ids)
            
        
    def register(self, server_address: str, expert_ids: List[int]):
        """
        Register a new server with the supported expert IDs.
        
        Args:
            server_address: The address of the server to register
            expert_ids: List of expert IDs supported by this server
        """
        # Add to server list if not already present
        if server_address not in self.server_addresses:
            self.server_addresses.append(server_address)
        
        # Update expert mapping
        for expert_id in expert_ids:
            if expert_id not in self.expert_to_servers:
                self.expert_to_servers[expert_id] = []
            if server_address not in self.expert_to_servers[expert_id]:
                self.expert_to_servers[expert_id].append(server_address)


    def unregister(self, server_address: str, expert_ids: Optional[List[int]] = None):
        """
            Unregister a server from the manager.
            
            Args:
            server_address: The address of the server to unregister
            expert_ids: Optional list of expert IDs to unregister for this server.
                    If None, all expert IDs associated with this server will be unregistered.
        """
        # If server not registered, nothing to do
        if server_address not in self.server_addresses:
            return
        
        # If expert_ids is None, unregister from all experts
        if expert_ids is None:
            self.server_addresses.remove(server_address)
            for expert_id, servers in list(self.expert_to_servers.items()):
                if server_address in servers:
                    servers.remove(server_address)
                    # If no servers left for this expert, remove the expert entry
                    if not servers:
                        del self.expert_to_servers[expert_id]
        else:
            # Unregister only specified expert IDs
            for expert_id in expert_ids:
                if expert_id in self.expert_to_servers and server_address in self.expert_to_servers[expert_id]:
                    self.expert_to_servers[expert_id].remove(server_address)
                    # If no servers left for this expert, remove the expert entry
                    if not self.expert_to_servers[expert_id]:
                        del self.expert_to_servers[expert_id]

    
    def choose_server_addresses(self, expert_ids: List[int]) -> List[str]:
        """
        Get all server addresses that support the specified expert IDs.
        
        Args:
            expert_ids: List of expert IDs to look up
            
        Returns:
            List of server addresses that support the specified expert IDs
        """
        # TODO: update policy
        
        return [self.choose_server_address(expert_id) for expert_id in expert_ids]


    def choose_server_address(self, expert_id: int) -> Optional[str]:
        """
        Get a server address that supports the specified expert ID.
        
        Args:
            expert_id: The expert ID to look up
            
        Returns:
            A server address, or None if no server supports this expert ID
        """
         # TODO: update policy

        if expert_id not in self.expert_to_servers or not self.expert_to_servers[expert_id]:
            return None
        
        # For now, just return a random server that supports this expert
        import random
        return random.choice(self.expert_to_servers[expert_id])