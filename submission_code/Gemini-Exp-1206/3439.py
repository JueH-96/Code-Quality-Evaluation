from collections import deque
from typing import List

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def get_tree_diameter_and_center(edges: List[List[int]]) -> tuple[int, int]:
            if not edges:
                return 0, 0
            n = 0
            for u, v in edges:
                n = max(n, u, v)
            n += 1
            graph = [[] for _ in range(n)]
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)

            def bfs(start: int) -> tuple[int, int]:
                dist = [-1] * n
                dist[start] = 0
                q = deque([start])
                farthest_node = start
                max_dist = 0
                while q:
                    u = q.popleft()
                    if dist[u] > max_dist:
                        max_dist = dist[u]
                        farthest_node = u
                    for v in graph[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            q.append(v)
                return farthest_node, max_dist

            farthest_node, _ = bfs(0)
            farthest_node, diameter = bfs(farthest_node)
            dist = [-1] * n
            dist[farthest_node] = 0
            q = deque([farthest_node])
            path = []
            while q:
                u = q.popleft()
                path.append(u)
                for v in graph[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            center = path[diameter // 2]
            return diameter, center

        diameter1, center1 = get_tree_diameter_and_center(edges1)
        diameter2, center2 = get_tree_diameter_and_center(edges2)
        return max(diameter1, diameter2, (diameter1 + 1) // 2 + (diameter2 + 1) // 2 + 1)