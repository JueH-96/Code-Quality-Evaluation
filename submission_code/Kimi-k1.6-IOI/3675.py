class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        # Sort edges by weight in descending order
        edges.sort(key=lambda x: -x[2])
        n = len(edges) + 1
        degree = [0] * n
        total = 0
        
        for u, v, w in edges:
            if degree[u] < k and degree[v] < k:
                total += w
                degree[u] += 1
                degree[v] += 1
        
        return total