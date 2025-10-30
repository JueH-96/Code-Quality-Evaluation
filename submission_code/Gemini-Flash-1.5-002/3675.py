class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = len(edges) + 1
        graph = [[] for _ in range(n)]
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))

        total_weight = 0
        for u, v, w in edges:
            total_weight += w

        edge_list = []
        for u, v, w in edges:
            edge_list.append((u, v, w))

        edge_list.sort(key=lambda x: x[2], reverse=True)

        degree = [0] * n
        selected_edges = []
        selected_weight = 0

        for u, v, w in edge_list:
            if degree[u] < k and degree[v] < k:
                degree[u] += 1
                degree[v] += 1
                selected_edges.append((u,v,w))
                selected_weight += w

        return selected_weight