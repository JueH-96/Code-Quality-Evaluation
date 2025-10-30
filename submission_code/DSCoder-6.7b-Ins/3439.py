from typing import List
from collections import defaultdict
import heapq

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        n = len(edges1) + 1
        m = len(edges2) + 1

        graph1 = defaultdict(list)
        graph2 = defaultdict(list)

        for u, v in edges1:
            graph1[u].append(v)
            graph1[v].append(u)

        for u, v in edges2:
            graph2[u].append(v)
            graph2[v].append(u)

        def dijkstra(graph, start):
            distances = [float('inf')] * n
            distances[start] = 0
            heap = [(0, start)]

            while heap:
                dist, node = heapq.heappop(heap)
                if distances[node] < dist:
                    continue
                for neighbor in graph[node]:
                    new_dist = dist + 1
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(heap, (new_dist, neighbor))
            return distances

        dist1 = dijkstra(graph1, 0)
        dist2 = dijkstra(graph2, 0)

        return max(dist1[i] + dist2[i] for i in range(n))