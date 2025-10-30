class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        from collections import defaultdict
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        # DFS with memoization
        # dp[node][parent_used] = max weight sum for subtree rooted at node
        # parent_used: 1 if edge to parent is used, 0 otherwise
        memo = {}
        
        def dfs(node, parent):
            if (node, parent) in memo:
                return memo[(node, parent)]
            
            # Get all children and their edge weights
            children_info = []
            for neighbor, weight in graph[node]:
                if neighbor != parent:
                    children_info.append((neighbor, weight))
            
            # For each child, calculate the benefit of including the edge to it
            child_benefits = []
            for child, weight in children_info:
                # Benefit = weight + max(child subtree with edge, child subtree without edge) - max(child subtree without edge)
                with_edge = weight + dfs(child, node)[1]  # child uses edge to parent (node)
                without_edge = dfs(child, node)[0]  # child doesn't use edge to parent
                benefit = with_edge - without_edge
                child_benefits.append((benefit, child, weight))
            
            # Sort by benefit in descending order
            child_benefits.sort(reverse=True)
            
            # Case 1: This node doesn't use edge to its parent
            # Can use up to k edges
            result_without_parent = 0
            edges_used = 0
            for benefit, child, weight in child_benefits:
                if edges_used < k and benefit > 0:
                    result_without_parent += weight + dfs(child, node)[1]
                    edges_used += 1
                else:
                    result_without_parent += dfs(child, node)[0]
            
            # Case 2: This node uses edge to its parent (if exists)
            # Can use up to k-1 additional edges
            result_with_parent = 0
            edges_used = 0
            available_slots = k - 1 if parent != -1 else k
            
            for benefit, child, weight in child_benefits:
                if edges_used < available_slots and benefit > 0:
                    result_with_parent += weight + dfs(child, node)[1]
                    edges_used += 1
                else:
                    result_with_parent += dfs(child, node)[0]
            
            memo[(node, parent)] = (result_without_parent, result_with_parent)
            return memo[(node, parent)]
        
        # Start DFS from node 0 (arbitrary root)
        return dfs(0, -1)[0]