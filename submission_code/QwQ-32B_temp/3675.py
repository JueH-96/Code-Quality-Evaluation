class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        # Sort edges in descending order of weight
        edges_sorted = sorted(edges, key=lambda x: -x[2])
        n = len(edges) + 1  # since there are n-1 edges in a tree
        degrees = [0] * n
        total = 0
        
        for u, v, w in edges_sorted:
            if degrees[u] < k and degrees[v] < k:
                total += w
                degrees[u] += 1
                degrees[v] += 1
        
        return total