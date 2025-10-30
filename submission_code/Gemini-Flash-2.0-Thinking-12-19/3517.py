class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        adj = [[] for _ in range(n)]
        for i in range(n - 1):
            adj[i].append(i + 1)

        def bfs(graph, start, end):
            dist = [-1] * n
            dist[start] = 0
            q = [start]
            while q:
                curr = q.pop(0)
                if curr == end:
                    return dist[end]
                for neighbor in graph[curr]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[curr] + 1
                        q.append(neighbor)
            return -1

        results = []
        for query in queries:
            u, v = query
            adj[u].append(v)
            shortest_path = bfs(adj, 0, n - 1)
            results.append(shortest_path)
        return results