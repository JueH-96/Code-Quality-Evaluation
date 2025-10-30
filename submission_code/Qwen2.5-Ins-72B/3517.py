from collections import defaultdict
import heapq

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        def dijkstra(graph, start, end):
            distances = [float('inf')] * n
            distances[start] = 0
            pq = [(0, start)]
            while pq:
                dist, node = heapq.heappop(pq)
                if dist > distances[node]:
                    continue
                for neighbor in graph[node]:
                    new_dist = dist + 1
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(pq, (new_dist, neighbor))
            return distances[end]
        
        graph = defaultdict(list)
        for i in range(n - 1):
            graph[i].append(i + 1)
        
        answer = []
        for u, v in queries:
            graph[u].append(v)
            answer.append(dijkstra(graph, 0, n - 1))
        
        return answer