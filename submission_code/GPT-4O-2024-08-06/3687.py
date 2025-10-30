from typing import List, Dict, Tuple
from collections import defaultdict

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        # Create adjacency list for the tree
        tree = defaultdict(list)
        for u, v, length in edges:
            tree[u].append((v, length))
            tree[v].append((u, length))
        
        # Result variables
        max_length = 0
        min_nodes = float('inf')
        
        # DFS function to explore the tree
        def dfs(node: int, parent: int, path_length: int, path_nodes: int, visited_values: set):
            nonlocal max_length, min_nodes
            
            # Add current node's value to visited set
            visited_values.add(nums[node])
            
            # Check if we have a new longest path
            if path_length > max_length:
                max_length = path_length
                min_nodes = path_nodes
            elif path_length == max_length:
                min_nodes = min(min_nodes, path_nodes)
            
            # Explore children
            for neighbor, edge_length in tree[node]:
                if neighbor != parent and nums[neighbor] not in visited_values:
                    dfs(neighbor, node, path_length + edge_length, path_nodes + 1, visited_values)
            
            # Backtrack: remove current node's value from visited set
            visited_values.remove(nums[node])
        
        # Start DFS from each node to find all possible special paths
        for i in range(len(nums)):
            dfs(i, -1, 0, 1, set())
        
        return [max_length, min_nodes]