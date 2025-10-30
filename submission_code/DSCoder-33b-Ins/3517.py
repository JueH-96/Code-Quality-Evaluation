class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        graph = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            graph[i][i] = 0
        res = []
        for u, v in queries:
            graph[u][v] = 1
            graph[v][u] = 1
            for i in range(n):
                for j in range(n):
                    graph[i][j] = min(graph[i][j], graph[i][u] + graph[u][v] + graph[v][j])
            res.append(graph[0][n - 1] if graph[0][n - 1] != float('inf') else -1)
        return res