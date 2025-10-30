import heapq
from typing import List

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append((v, 2 if v % 2 == 0 else 1))
            adj[v].append((u, 2 if u % 2 == 0 else 1))

        def dijkstra(start):
            dist = [float('inf')] * n
            dist[start] = 0
            heap = []
            heapq.heappush(heap, (0, start))
            while heap:
                d, u = heapq.heappop(heap)
                if d > dist[u]:
                    continue
                for v, w in adj[u]:
                    if dist[v] > dist[u] + w:
                        dist[v] = dist[u] + w
                        heapq.heappush(heap, (dist[v], v))
            max_dist = max(dist)
            farthest = dist.index(max_dist)
            return dist, farthest

        # Find first farthest node
        _, u = dijkstra(0)
        # Find second farthest node and distances from u
        dist_u, v = dijkstra(u)
        # Find distances from v
        dist_v, _ = dijkstra(v)

        # Compute the answer for each node
        ans = []
        for i in range(n):
            max_time = max(dist_u[i], dist_v[i])
            ans.append(max_time)
        
        return ans