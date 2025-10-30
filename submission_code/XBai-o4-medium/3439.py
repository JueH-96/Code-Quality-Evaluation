from collections import deque
from typing import List

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def compute_diameter(edges):
            n = len(edges) + 1
            if n == 0:
                return 0
            adj = [[] for _ in range(n)]
            for a, b in edges:
                adj[a].append(b)
                adj[b].append(a)
            
            def bfs(start):
                q = deque()
                q.append((start, -1, 0))
                max_dist = 0
                farthest_node = start
                while q:
                    node, parent, dist = q.popleft()
                    if dist > max_dist:
                        max_dist = dist
                        farthest_node = node
                    for neighbor in adj[node]:
                        if neighbor != parent:
                            q.append((neighbor, node, dist + 1))
                return farthest_node, max_dist
            
            u, _ = bfs(0)
            v, diameter = bfs(u)
            return diameter
        
        d1 = compute_diameter(edges1)
        d2 = compute_diameter(edges2)
        
        candidate = ((d1 + 1) // 2) + ((d2 + 1) // 2) + 1
        
        return max(d1, d2, candidate)