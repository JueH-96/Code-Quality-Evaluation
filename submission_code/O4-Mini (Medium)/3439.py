from collections import deque
from typing import List

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        # Helper to compute diameter and radius of a tree given its edges.
        def tree_diameter_and_radius(edges: List[List[int]]) -> (int, int):
            n = len(edges) + 1
            # Build adjacency list
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # BFS from a start node to find farthest node and distances
            def bfs(start: int) -> (int, List[int]):
                dist = [-1] * n
                dist[start] = 0
                q = deque([start])
                far_node = start
                while q:
                    u = q.popleft()
                    # Track the farthest node seen so far
                    if dist[u] > dist[far_node]:
                        far_node = u
                    for w in adj[u]:
                        if dist[w] == -1:
                            dist[w] = dist[u] + 1
                            q.append(w)
                return far_node, dist
            
            # 1) From node 0 find a farthest node u
            u, _ = bfs(0)
            # 2) From u find the farthest node v and record distances du
            v, du = bfs(u)
            diameter = du[v]
            # 3) From v record distances dv
            _, dv = bfs(v)
            # 4) Radius is min over all nodes of max(du[i], dv[i])
            radius = min(max(du[i], dv[i]) for i in range(n))
            return diameter, radius
        
        # Compute diameter and radius for both trees
        D1, R1 = tree_diameter_and_radius(edges1)
        D2, R2 = tree_diameter_and_radius(edges2)
        
        # The best merge connects the centers, giving:
        # new_diameter = max(D1, D2, R1 + 1 + R2)
        return max(D1, D2, R1 + 1 + R2)