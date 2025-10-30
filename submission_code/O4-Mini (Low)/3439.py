from collections import deque
from typing import List

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        # Helper to compute the diameter of a tree given its edges and node count
        def tree_diameter_and_radius(n: int, edges: List[List[int]]) -> (int, int):
            if n == 1:
                return 0, 0  # single node: diameter 0, radius 0
            
            # Build adjacency
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # BFS to find farthest from an arbitrary start (0)
            def bfs(start: int):
                dist = [-1] * n
                dq = deque([start])
                dist[start] = 0
                far_node = start
                while dq:
                    u = dq.popleft()
                    for w in adj[u]:
                        if dist[w] == -1:
                            dist[w] = dist[u] + 1
                            dq.append(w)
                            if dist[w] > dist[far_node]:
                                far_node = w
                return far_node, dist
            
            # First BFS to get one end of diameter
            u, _ = bfs(0)
            # Second BFS from u to get the diameter length
            v, dist_u = bfs(u)
            diameter = dist_u[v]
            # Radius is ceil(diameter / 2)
            radius = (diameter + 1) // 2
            return diameter, radius
        
        n = len(edges1) + 1
        m = len(edges2) + 1
        
        d1, r1 = tree_diameter_and_radius(n, edges1)
        d2, r2 = tree_diameter_and_radius(m, edges2)
        
        # When we connect the two trees optimally, the best is to link their centers:
        # The resulting diameter is the max of:
        #   the larger original diameter, and the path going center1 -> new edge -> center2
        return max(max(d1, d2), r1 + 1 + r2)