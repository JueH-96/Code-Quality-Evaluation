class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        from collections import defaultdict
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, length in edges:
            graph[u].append((v, length))
            graph[v].append((u, length))
        
        n = len(nums)
        max_length = 0
        min_nodes = float('inf')
        
        def dfs(node, parent, visited_values, current_length, node_count):
            nonlocal max_length, min_nodes
            
            # Update result if we found a longer path or same length with fewer nodes
            if current_length > max_length:
                max_length = current_length
                min_nodes = node_count
            elif current_length == max_length:
                min_nodes = min(min_nodes, node_count)
            
            # Try to extend the path to children
            for neighbor, edge_length in graph[node]:
                if neighbor != parent and nums[neighbor] not in visited_values:
                    visited_values.add(nums[neighbor])
                    dfs(neighbor, node, visited_values, current_length + edge_length, node_count + 1)
                    visited_values.remove(nums[neighbor])
        
        # Start DFS from each node
        for start_node in range(n):
            visited = {nums[start_node]}
            dfs(start_node, -1, visited, 0, 1)
        
        return [max_length, min_nodes]