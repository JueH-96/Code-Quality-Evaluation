from typing import List

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = len(edges) + 1
        degrees = [0] * n
        total = 0
        
        # Calculate initial degrees and total sum
        for u, v, w in edges:
            degrees[u] += 1
            degrees[v] += 1
            total += w
        
        # Sort edges by weight in ascending order
        edges_sorted = sorted(edges, key=lambda x: x[2])
        
        # Iterate through sorted edges and remove if necessary
        for u, v, w in edges_sorted:
            if degrees[u] > k or degrees[v] > k:
                total -= w
                degrees[u] -= 1
                degrees[v] -= 1
        
        return total