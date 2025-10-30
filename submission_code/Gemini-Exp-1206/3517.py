from typing import List
import heapq

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        answer = []
        edges = []
        for i in range(n - 1):
            edges.append((i, i + 1))

        for i in range(len(queries)):
            edges.append((queries[i][0], queries[i][1]))
            graph = {j: [] for j in range(n)}
            for u, v in edges:
                graph[u].append(v)

            dist = {j: float('inf') for j in range(n)}
            dist[0] = 0
            pq = [(0, 0)]

            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                for v in graph[u]:
                    if dist[v] > dist[u] + 1:
                        dist[v] = dist[u] + 1
                        heapq.heappush(pq, (dist[v], v))

            answer.append(dist[n - 1])

        return answer