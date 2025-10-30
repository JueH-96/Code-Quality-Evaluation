class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        # Sort the edges in descending order based on their weights
        edges_sorted = sorted(edges, key=lambda x: -x[2])
        n = len(edges) + 1  # Number of nodes is edges + 1
        degrees = [0] * n  # Track the degree of each node
        total = 0
        
        for u, v, w in edges_sorted:
            if degrees[u] < k and degrees[v] < k:
                total += w
                degrees[u] += 1
                degrees[v] += 1
        
        return total