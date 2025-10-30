from collections import deque
from typing import List

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def get_diameter_and_radius(edges: List[List[int]]) -> (int, int):
            n = len(edges) + 1
            # build adjacency list
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            # 1) BFS from node 0 to find farthest node A
            dist = [-1] * n
            dist[0] = 0
            q = deque([0])
            while q:
                x = q.popleft()
                for y in adj[x]:
                    if dist[y] < 0:
                        dist[y] = dist[x] + 1
                        q.append(y)
            A = max(range(n), key=lambda i: dist[i])
            # 2) BFS from A to find farthest node B and distances distA
            distA = [-1] * n
            distA[A] = 0
            q = deque([A])
            while q:
                x = q.popleft()
                for y in adj[x]:
                    if distA[y] < 0:
                        distA[y] = distA[x] + 1
                        q.append(y)
            B = max(range(n), key=lambda i: distA[i])
            diameter = distA[B]
            # radius = ceil(diameter / 2)
            radius = (diameter + 1) // 2
            return diameter, radius

        D1, r1 = get_diameter_and_radius(edges1)
        D2, r2 = get_diameter_and_radius(edges2)
        # best merge connects centers of the two trees
        return max(D1, D2, r1 + 1 + r2)