class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        from collections import defaultdict
        
        n = len(nums)
        if n == 1:
            return [0, 1]
        
        # Build adjacency list with weights
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        # Find root (node 0) and build parent relationships
        parent = [-1] * n
        edge_weight = {}
        
        def build_tree(node, par):
            parent[node] = par
            for neighbor, weight in graph[node]:
                if neighbor != par:
                    edge_weight[(node, neighbor)] = weight
                    build_tree(neighbor, node)
        
        build_tree(0, -1)
        
        max_length = 0
        min_nodes = float('inf')
        
        def dfs(node):
            nonlocal max_length, min_nodes
            
            # For each child, find the longest path going down
            child_paths = []  # (length, nodes, set of values)
            
            for neighbor, weight in graph[node]:
                if neighbor != parent[node]:  # Only go to children
                    child_length, child_nodes, child_values = dfs(neighbor)
                    if nums[node] not in child_values:
                        # Can extend this path
                        new_values = child_values.copy()
                        new_values.add(nums[node])
                        child_paths.append((child_length + weight, child_nodes + 1, new_values))
            
            # Path starting at this node
            current_path = (0, 1, {nums[node]})
            
            # Update global max with just this node
            if 0 > max_length:
                max_length = 0
                min_nodes = 1
            elif 0 == max_length:
                min_nodes = min(min_nodes, 1)
            
            # Check paths going through this node
            all_paths = child_paths + [current_path]
            
            # Check single paths from this node
            for length, nodes, values in all_paths:
                if length > max_length:
                    max_length = length
                    min_nodes = nodes
                elif length == max_length:
                    min_nodes = min(min_nodes, nodes)
            
            # Check pairs of paths (going through this node)
            for i in range(len(child_paths)):
                for j in range(i + 1, len(child_paths)):
                    length1, nodes1, values1 = child_paths[i]
                    length2, nodes2, values2 = child_paths[j]
                    
                    # Check if paths can be combined (no common values except current node)
                    if not (values1 & values2):
                        total_length = length1 + length2
                        total_nodes = nodes1 + nodes2 - 1  # -1 because we count current node once
                        
                        if total_length > max_length:
                            max_length = total_length
                            min_nodes = total_nodes
                        elif total_length == max_length:
                            min_nodes = min(min_nodes, total_nodes)
            
            # Return the best path going down from this node
            if child_paths:
                return max(child_paths, key=lambda x: x[0])
            else:
                return current_path
        
        dfs(0)
        
        return [max_length, min_nodes]