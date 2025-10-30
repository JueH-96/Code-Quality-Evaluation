from typing import List
from collections import deque

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def get_diameter_and_radius(edges):
            if not edges:
                return (0, 0)
            
            n = len(edges) + 1
            adj = [[] for _ in range(n)]
            for a, b in edges:
                adj[a].append(b)
                adj[b].append(a)
            
            def bfs(start):
                q = deque([start])
                dist = [-1] * n
                dist[start] = 0
                while q:
                    node = q.popleft()
                    for neighbor in adj[node]:
                        if dist[neighbor] == -1:
                            dist[neighbor] = dist[node] + 1
                            q.append(neighbor)
                return dist
            
            # First BFS to find the farthest node from node 0 (or any node)
            dist0 = bfs(0)
            u = dist0.index(max(dist0))
            
            # Second BFS to find the farthest node from u and get the diameter
            dist_u = bfs(u)
            v = dist_u.index(max(dist_u))
            diameter = max(dist_u)
            
            # Third BFS to get distances from v
            dist_v = bfs(v)
            
            # Calculate the radius as the minimum of the maximum distances from u and v
            radius = min(max(du, dv) for du, dv in zip(dist_u, dist_v))
            
            return (diameter, radius)
        
        d1, r1 = get_diameter_and_radius(edges1)
        d2, r2 = get_diameter_and_radius(edges2)
        
        return max(d1, d2, r1 + r2 + 1)