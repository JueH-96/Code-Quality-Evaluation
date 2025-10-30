class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        edges.sort(key=lambda x: x[2], reverse=True)
        n = 0
        for u, v, _ in edges:
            n = max(n, u, v)
        n += 1
        degrees = [0] * n
        max_weight_sum = 0

        selected_edges = []

        for u, v, weight in edges:
            if degrees[u] < k and degrees[v] < k:
                selected_edges.append((u, v, weight))
                degrees[u] += 1
                degrees[v] += 1
                max_weight_sum += weight

        return max_weight_sum