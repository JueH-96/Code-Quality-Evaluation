class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = len(edges) + 1
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        
        # DFS to calculate maximum weight
        def dfs(node, parent):
            # For each child, calculate:
            # - max weight if we include the edge to this child
            # - max weight if we don't include the edge to this child
            children_gains = []
            base_sum = 0
            
            for child, weight in adj[node]:
                if child == parent:
                    continue
                
                # Recursively calculate for child subtree
                include_child, exclude_child = dfs(child, node)
                
                # If we include edge to child, child can use at most k-1 other edges
                gain = weight + include_child - exclude_child
                base_sum += exclude_child
                
                if gain > 0:
                    children_gains.append(gain)
            
            # Sort gains in descending order
            children_gains.sort(reverse=True)
            
            # Calculate max weight if parent edge is included (can use k-1 children edges)
            include_parent = base_sum + sum(children_gains[:k-1])
            
            # Calculate max weight if parent edge is not included (can use k children edges)
            exclude_parent = base_sum + sum(children_gains[:k])
            
            return include_parent, exclude_parent
        
        # Start DFS from node 0
        _, result = dfs(0, -1)
        return result