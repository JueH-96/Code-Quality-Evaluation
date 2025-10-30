class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        from collections import defaultdict, deque
        
        # Build the graph from the edges
        graph = defaultdict(list)
        for u, v, length in edges:
            graph[u].append((v, length))
            graph[v].append((u, length))
        
        # DFS function to find the longest special path
        def dfs(node, parent, value_set, current_length):
            nonlocal max_length, min_nodes
            # Check if current node value is already in the set (not unique)
            if nums[node] in value_set:
                return
            
            # Add current node value to the set
            new_value_set = value_set | {nums[node]}
            
            # Update the longest special path found
            if current_length > max_length:
                max_length = current_length
                min_nodes = len(new_value_set)
            elif current_length == max_length:
                min_nodes = min(min_nodes, len(new_value_set))
            
            # Visit all adjacent nodes
            for neighbor, length in graph[node]:
                if neighbor != parent:
                    dfs(neighbor, node, new_value_set, current_length + length)
        
        max_length = 0
        min_nodes = float('inf')
        
        # Start DFS from the root node
        dfs(0, -1, set(), 0)
        
        return [max_length, min_nodes]