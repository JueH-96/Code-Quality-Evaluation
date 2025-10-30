from collections import deque
from typing import List

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        """
        The minimum diameter after merging two trees equals
            max(d1, d2, r1 + 1 + r2)
        where
            d1, d2 – the diameters of the two trees
            r1, r2 – their radii  (radius = ceil(diameter / 2))
        The value r1 + 1 + r2 is achieved by linking a centre of the first
        tree with a centre of the second tree, hence the formula gives the
        optimum.
        """

        def tree_diameter(edges: List[List[int]]) -> int:
            """Return the diameter (longest distance in edges) of a tree."""
            n = len(edges) + 1          # number of vertices

            # adjacency list
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)

            if n == 1:                   # single-vertex tree
                return 0

            # helper bfs – returns (farthest_vertex, distance_to_it)
            def bfs(start: int):
                dist = [-1] * n
                dist[start] = 0
                q = deque([start])
                far = start
                while q:
                    node = q.popleft()
                    for nb in adj[node]:
                        if dist[nb] == -1:
                            dist[nb] = dist[node] + 1
                            far = nb
                            q.append(nb)
                return far, dist[far]

            u, _ = bfs(0)        # one end of the diameter
            _, d = bfs(u)        # diameter length
            return d

        # diameters of both trees
        d1 = tree_diameter(edges1)
        d2 = tree_diameter(edges2)

        # radii of both trees  (ceil(d / 2))
        r1 = (d1 + 1) // 2
        r2 = (d2 + 1) // 2

        # minimal possible diameter after adding one edge
        return max(d1, d2, r1 + 1 + r2)