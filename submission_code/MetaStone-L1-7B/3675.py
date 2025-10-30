class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = len(edges)
        edges_sorted = sorted(edges, key=lambda x: -x[2])
        counts = [0] * n
        sum_total = 0
        
        for u, v, w in edges_sorted:
            if counts[u] < k and counts[v] < k:
                sum_total += w
                counts[u] += 1
                counts[v] += 1
        
        return sum_total