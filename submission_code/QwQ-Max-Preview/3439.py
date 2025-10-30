from typing import List
from collections import deque

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def compute_diameter(edges):
            if not edges:
                return 0  # Single node tree has diameter 0
            n = len(edges) + 1
            adj = [[] for _ in range(n)]
            for a, b in edges:
                adj[a].append(b)
                adj[b].append(a)
            
            def bfs(start):
                visited = [-1] * n
                q = deque()
                q.append(start)
                visited[start] = 0
                max_dist = 0
                farthest_node = start
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if visited[v] == -1:
                            visited[v] = visited[u] + 1
                            q.append(v)
                            if visited[v] > max_dist:
                                max_dist = visited[v]
                                farthest_node = v
                return farthest_node, max_dist
            
            u, _ = bfs(0)
            v, diameter = bfs(u)
            return diameter
        
        d1 = compute_diameter(edges1)
        d2 = compute_diameter(edges2)
        r1 = (d1 + 1) // 2
        r2 = (d2 + 1) // 2
        candidate = r1 + r2 + 1
        return max(d1, d2, candidate)