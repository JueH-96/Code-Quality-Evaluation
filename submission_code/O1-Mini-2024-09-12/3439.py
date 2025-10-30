from typing import List
from collections import deque

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def build_adj(n, edges):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            return adj
        
        def bfs(adj, start):
            n = len(adj)
            visited = [False]*n
            queue = deque()
            queue.append((start, 0))
            visited[start] = True
            farthest_node = start
            max_dist = 0
            while queue:
                node, dist = queue.popleft()
                if dist > max_dist:
                    max_dist = dist
                    farthest_node = node
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append((neighbor, dist +1))
            return farthest_node, max_dist
        
        def get_diameter_and_radius(edges, n):
            adj = build_adj(n, edges)
            far_node, _ = bfs(adj, 0)
            far_node2, diameter = bfs(adj, far_node)
            radius = (diameter +1)//2
            return diameter, radius
        
        n = len(edges1) +1
        m = len(edges2) +1
        d1, r1 = get_diameter_and_radius(edges1, n)
        d2, r2 = get_diameter_and_radius(edges2, m)
        return max(d1, d2, r1 + r2 +1)