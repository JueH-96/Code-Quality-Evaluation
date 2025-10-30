class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        # Sort edges by weight in descending order
        edges.sort(key=lambda x: x[2], reverse=True)
        
        # Track the degree of each node
        from collections import defaultdict
        degree = defaultdict(int)
        
        max_sum = 0
        
        # Greedily add edges from largest to smallest weight
        for u, v, w in edges:
            # If adding this edge does not exceed the degree limit for either endpoint, add it
            if degree[u] < k and degree[v] < k:
                max_sum += w
                degree[u] += 1
                degree[v] += 1
        
        return max_sum