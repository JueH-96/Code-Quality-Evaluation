from typing import List
import heapq

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize the graph with the initial roads
        graph = {i: [] for i in range(n)}
        for i in range(n - 1):
            graph[i].append((i + 1, 1))

        def dijkstra(start, end):
            # Priority queue to store (distance, node)
            pq = [(0, start)]
            # Distance dictionary to store the shortest distance to each node
            dist = {i: float('inf') for i in range(n)}
            dist[start] = 0

            while pq:
                current_dist, u = heapq.heappop(pq)

                if u == end:
                    return current_dist

                if current_dist > dist[u]:
                    continue

                for v, weight in graph[u]:
                    distance = current_dist + weight

                    if distance < dist[v]:
                        dist[v] = distance
                        heapq.heappush(pq, (distance, v))

            return dist[end]

        result = []
        for u, v in queries:
            # Add the new road to the graph
            graph[u].append((v, 1))
            # Calculate the shortest path from 0 to n-1
            shortest_path = dijkstra(0, n - 1)
            result.append(shortest_path)

        return result