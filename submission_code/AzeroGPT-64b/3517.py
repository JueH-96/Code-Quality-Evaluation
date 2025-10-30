class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        graph = defaultdict(list)
        dist = [float(inf)] * n
        dist[0] = 0

        for q in queries:
            u, v = q[0], q[1]
            graph[u].append(v)

        for i in range(len(queries)):
            if queries[i][0] == 0:
                dist[queries[i][1]] = 1

            for j in range(n):
                if dist[j] == float(inf): continue
                for k in graph[j]:
                    dist[k] = min(dist[k], dist[j] + 1)

            yield dist[n - 1]