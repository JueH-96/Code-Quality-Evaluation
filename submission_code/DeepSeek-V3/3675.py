from collections import defaultdict

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        # Build adjacency list
        adj = defaultdict(list)
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        
        # Initialize degree count for each node
        degree = defaultdict(int)
        for u, v, _ in edges:
            degree[u] += 1
            degree[v] += 1
        
        # Sort edges by weight in descending order
        edges_sorted = sorted(edges, key=lambda x: -x[2])
        
        # Initialize the sum of weights
        total_weight = 0
        
        # Iterate through sorted edges and add to the sum if possible
        for u, v, w in edges_sorted:
            if degree[u] > k and degree[v] > k:
                # Remove the edge with the smallest weight among the edges connected to u or v
                # Since edges are sorted, the current edge is the smallest among the remaining
                # So, we can safely remove it
                degree[u] -= 1
                degree[v] -= 1
            else:
                total_weight += w
        
        return total_weight