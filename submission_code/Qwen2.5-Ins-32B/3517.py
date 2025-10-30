from typing import List
import heapq

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        def dijkstra(graph, start, end):
            dist = [float('inf')] * n
            dist[start] = 0
            pq = [(0, start)]
            while pq:
                d, node = heapq.heappop(pq)
                if node == end:
                    return d
                if d > dist[node]:
                    continue
                for neighbor in graph[node]:
                    if dist[node] + 1 < dist[neighbor]:
                        dist[neighbor] = dist[node] + 1
                        heapq.heappush(pq, (dist[neighbor], neighbor))
            return dist[end]
        
        graph = {i: [i + 1] for i in range(n - 1)}
        graph[n - 1] = []
        answer = []
        
        for u, v in queries:
            graph[u].append(v)
            answer.append(dijkstra(graph, 0, n - 1))
        
        return answer