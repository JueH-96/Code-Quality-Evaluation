class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        from collections import defaultdict
        
        # Build the tree
        graph = defaultdict(list)
        for u, v, length in edges:
            graph[u].append((v, length))
            graph[v].append((u, length))
        
        n = len(nums)
        max_length = 0
        min_nodes = float('inf')
        
        # Build rooted tree with parent relationships
        children = defaultdict(list)
        visited = [False] * n
        
        def build_tree(node):
            visited[node] = True
            for neighbor, edge_length in graph[node]:
                if not visited[neighbor]:
                    children[node].append((neighbor, edge_length))
                    build_tree(neighbor)
        
        build_tree(0)
        
        def dfs(node, current_length, current_nodes, seen_values):
            nonlocal max_length, min_nodes
            
            # Update the result
            if current_length > max_length:
                max_length = current_length
                min_nodes = current_nodes
            elif current_length == max_length:
                min_nodes = min(min_nodes, current_nodes)
            
            # Explore children
            for child, edge_length in children[node]:
                if nums[child] not in seen_values:
                    seen_values.add(nums[child])
                    dfs(child, current_length + edge_length, current_nodes + 1, seen_values)
                    seen_values.remove(nums[child])
        
        # Try starting from each node
        for start in range(n):
            seen = {nums[start]}
            dfs(start, 0, 1, seen)
        
        return [max_length, min_nodes]