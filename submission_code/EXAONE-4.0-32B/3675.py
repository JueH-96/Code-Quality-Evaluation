from typing import List

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = len(edges) + 1
        deg = [0] * n
        edges.sort(key=lambda x: x[2], reverse=True)
        total = 0
        for u, v, w in edges:
            if deg[u] < k and deg[v] < k:
                total += w
                deg[u] += 1
                deg[v] += 1
        return total