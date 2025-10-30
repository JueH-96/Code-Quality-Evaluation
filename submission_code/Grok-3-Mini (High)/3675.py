from typing import List

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = len(edges) + 1
        deg = [0] * n
        sorted_edges = sorted(edges, key=lambda x: x[2], reverse=True)
        total_sum = 0
        for edge in sorted_edges:
            u, v, w = edge
            if deg[u] < k and deg[v] < k:
                total_sum += w
                deg[u] += 1
                deg[v] += 1
        return total_sum