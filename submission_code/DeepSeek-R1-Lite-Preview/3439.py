from typing import List
from collections import deque

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def find_diameter(edges, n):
            if n == 1:
                return 0
            adj = [[] for _ in range(n)]
            for a, b in edges:
                adj[a].append(b)
                adj[b].append(a)
            
            def bfs(start):
                visited = [False] * n
                distance = [0] * n
                queue = deque([start])
                visited[start] = True
                while queue:
                    u = queue.popleft()
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            distance[v] = distance[u] + 1
                            queue.append(v)
                farthest_node = start
                max_distance = 0
                for i in range(n):
                    if distance[i] > max_distance:
                        max_distance = distance[i]
                        farthest_node = i
                return farthest_node, max_distance
            
            u, _ = bfs(0)
            v, diameter = bfs(u)
            return diameter
        
        n = len(edges1) + 1
        m = len(edges2) + 1
        
        D_A = find_diameter(edges1, n)
        D_B = find_diameter(edges2, m)
        
        depth_A = (D_A + 1) // 2
        depth_B = (D_B + 1) // 2
        
        minimal_diameter = max(D_A, D_B, depth_A + depth_B + 1)
        
        return minimal_diameter