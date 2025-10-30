from collections import defaultdict
from typing import List

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        # Build the graph from edges
        graph = defaultdict(list)
        for u, v, length in edges:
            graph[u].append((v, length))
            graph[v].append((u, length))
        
        # Variables to store the result
        max_length = 0
        min_nodes = float('inf')
        
        def dfs(node, parent, current_length, visited):
            nonlocal max_length, min_nodes
            
            # Check if the current node's value is already visited
            if nums[node] in visited:
                return
            
            # Mark the current node as visited
            visited.add(nums[node])
            
            # Update the max_length and min_nodes
            if current_length > max_length:
                max_length = current_length
                min_nodes = 1  # Reset to 1 because we found a longer path
            elif current_length == max_length:
                min_nodes = min(min_nodes, len(visited))
            
            # Explore the neighbors
            for neighbor, length in graph[node]:
                if neighbor != parent:  # Avoid going back to the parent
                    dfs(neighbor, node, current_length + length, visited)
            
            # Backtrack: unmark the current node's value
            visited.remove(nums[node])
        
        # Start DFS from the root node (0)
        dfs(0, -1, 0, set())
        
        return [max_length, min_nodes]