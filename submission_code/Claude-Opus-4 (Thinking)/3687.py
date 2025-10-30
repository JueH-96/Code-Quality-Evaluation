class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        n = len(nums)
        
        # Build adjacency list with edge lengths
        adj = [[] for _ in range(n)]
        for u, v, length in edges:
            adj[u].append((v, length))
            adj[v].append((u, length))
        
        # Build tree structure with root 0
        children = [[] for _ in range(n)]
        edge_lengths = {}
        
        visited = [False] * n
        def build_tree(node):
            visited[node] = True
            for neighbor, length in adj[node]:
                if not visited[neighbor]:
                    children[node].append(neighbor)
                    edge_lengths[(node, neighbor)] = length
                    build_tree(neighbor)
        
        build_tree(0)
        
        # Find longest special path
        max_length = 0
        min_nodes = 1  # At least one node in any path
        
        def dfs(node, used_values, path_length, path_nodes):
            nonlocal max_length, min_nodes
            
            # Update result if we found a better or equal path
            if path_length > max_length:
                max_length = path_length
                min_nodes = path_nodes
            elif path_length == max_length:
                min_nodes = min(min_nodes, path_nodes)
            
            # Try extending path to each child
            for child in children[node]:
                if nums[child] not in used_values:
                    used_values.add(nums[child])
                    dfs(child, used_values, path_length + edge_lengths[(node, child)], path_nodes + 1)
                    used_values.remove(nums[child])
        
        # Start DFS from each node as potential starting point
        for start in range(n):
            used = {nums[start]}
            dfs(start, used, 0, 1)
        
        return [max_length, min_nodes]