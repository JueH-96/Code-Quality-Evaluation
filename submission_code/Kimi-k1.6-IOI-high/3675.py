class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        # Sort edges by weight in descending order
        edges.sort(key=lambda x: -x[2])
        max_node = 0
        for u, v, _ in edges:
            max_node = max(max_node, u, v)
        n = max_node + 1  # Determine the number of nodes dynamically
        degree = [0] * n
        total = 0
        for u, v, w in edges:
            if degree[u] < k and degree[v] < k:
                total += w
                degree[u] += 1
                degree[v] += 1
        return total