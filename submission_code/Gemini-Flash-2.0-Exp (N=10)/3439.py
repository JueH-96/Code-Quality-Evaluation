from collections import defaultdict, deque
from typing import List

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def bfs(graph, start):
            distances = [-1] * len(graph)
            distances[start] = 0
            queue = deque([start])
            farthest_node = start
            max_dist = 0

            while queue:
                node = queue.popleft()
                for neighbor in graph[node]:
                    if distances[neighbor] == -1:
                        distances[neighbor] = distances[node] + 1
                        queue.append(neighbor)
                        if distances[neighbor] > max_dist:
                            max_dist = distances[neighbor]
                            farthest_node = neighbor
            return farthest_node, max_dist

        def get_diameter(edges, num_nodes):
            graph = defaultdict(list)
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)
            
            if not graph:
                return 0
            
            farthest_node, _ = bfs(graph, 0)
            _, diameter = bfs(graph, farthest_node)
            return diameter

        n = len(edges1) + 1
        m = len(edges2) + 1

        diameter1 = get_diameter(edges1, n)
        diameter2 = get_diameter(edges2, m)

        
        graph1 = defaultdict(list)
        for u, v in edges1:
            graph1[u].append(v)
            graph1[v].append(u)
        
        graph2 = defaultdict(list)
        for u, v in edges2:
            graph2[u].append(v)
            graph2[v].append(u)

        
        max_radius1 = 0
        for i in range(n):
            _, max_dist = bfs(graph1, i)
            max_radius1 = max(max_radius1, (max_dist + 1) // 2)
        
        max_radius2 = 0
        for i in range(m):
            _, max_dist = bfs(graph2, i)
            max_radius2 = max(max_radius2, (max_dist + 1) // 2)

        return max(diameter1, diameter2, max_radius1 + max_radius2 + 1)