from collections import deque
from typing import List

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def compute_diameter(n: int, edges: List[List[int]]) -> int:
            if n == 1:
                return 0
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            def bfs(start):
                dist = [-1] * n
                q = deque([start])
                dist[start] = 0
                max_dist = 0
                far_node = start

                while q:
                    node = q.popleft()
                    for nei in adj[node]:
                        if dist[nei] == -1:
                            dist[nei] = dist[node] + 1
                            if dist[nei] > max_dist:
                                max_dist = dist[nei]
                                far_node = nei
                            q.append(nei)
                return far_node, dist
            
            u, _ = bfs(0)
            v, dist_v = bfs(u)
            return dist_v[v]
        
        n = len(edges1) + 1
        m = len(edges2) + 1

        diam1 = compute_diameter(n, edges1)
        diam2 = compute_diameter(m, edges2)

        r1 = (diam1 + 1) // 2
        r2 = (diam2 + 1) // 2

        candidate = r1 + r2 + 1

        return max(diam1, diam2, candidate)