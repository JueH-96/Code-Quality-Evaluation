from collections import defaultdict

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        # Sort edges by weight in ascending order
        sorted_edges = sorted(edges, key=lambda x: x[2])
        
        # Keep track of how many neighbors each node has
        neighbors_count = defaultdict(int)
        for u, v, _ in edges:
            neighbors_count[u] += 1
            neighbors_count[v] += 1
        
        # Calculate total weight of all edges
        total_weight = sum(w for _, _, w in edges)
        removed_weight = 0
        
        # Greedily remove edges with smallest weights until all nodes have ≤ k neighbors
        for u, v, w in sorted_edges:
            # Only remove if either node has more than k neighbors
            if neighbors_count[u] > k or neighbors_count[v] > k:
                removed_weight += w
                neighbors_count[u] -= 1
                neighbors_count[v] -= 1
        
        return total_weight - removed_weight