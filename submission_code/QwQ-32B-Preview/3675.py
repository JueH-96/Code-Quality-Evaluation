from typing import List

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = len(edges) + 1  # Calculate number of nodes
        # Sort edges in descending order of weight
        sorted_edges = sorted(edges, key=lambda x: x[2], reverse=True)
        
        degrees = [0] * n  # Initialize degrees array
        sum_weights = 0    # Initialize sum of weights
        
        # Iterate through sorted edges
        for edge in sorted_edges:
            u, v, w = edge
            if degrees[u] < k and degrees[v] < k:
                degrees[u] += 1
                degrees[v] += 1
                sum_weights += w
        
        return sum_weights