class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        n = len(nums)
        # Create adjacency list representation of the tree
        graph = [[] for _ in range(n)]
        for u, v, length in edges:
            graph[u].append((v, length))
            graph[v].append((u, length))
        
        # Variables to store the result
        max_length = 0
        min_nodes = float('inf')
        
        def dfs(node, parent, path_length, seen_values):
            nonlocal max_length, min_nodes
            
            # Count nodes in current path
            current_nodes = len(seen_values)
            
            # If we found a longer path
            if path_length > max_length:
                max_length = path_length
                min_nodes = current_nodes
            # If we found a path of same length
            elif path_length == max_length:
                min_nodes = min(min_nodes, current_nodes)
            
            # Try all neighbors
            for neighbor, length in graph[node]:
                if neighbor != parent and nums[neighbor] not in seen_values:
                    # Add current value to seen set and explore
                    seen_values.add(nums[neighbor])
                    dfs(neighbor, node, path_length + length, seen_values)
                    # Remove current value from seen set (backtrack)
                    seen_values.remove(nums[neighbor])
        
        # Try starting DFS from each node
        for start in range(n):
            seen = {nums[start]}
            dfs(start, -1, 0, seen)
        
        # If no valid path found, return [0, 1]
        if max_length == 0:
            return [0, 1]
            
        return [max_length, min_nodes]