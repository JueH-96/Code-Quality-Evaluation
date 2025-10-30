from typing import List
import heapq

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        graph = [[] for _ in range(n)]
        for i in range(n - 1):
            graph[i].append((i + 1, 1))
        
        for u, v in queries:
            graph[u].append((v, 1))
        
        def dijkstra():
            dist = [float('inf')] * n
            dist[0] = 0
            pq = [(0, 0)]
            while pq:
                d, node = heapq.heappop(pq)
                if d > dist[node]:
                    continue
                for neighbor, weight in graph[node]:
                    if dist[neighbor] > d + weight:
                        dist[neighbor] = d + weight
                        heapq.heappush(pq, (d + weight, neighbor))
            return dist[-1]
        
        result = []
        for _ in range(len(queries)):
            result.append(dijkstra())
            u, v = queries[_]
            graph[u].append((v, 1))
        
        return result