from collections import defaultdict, deque
from typing import List

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def build_graph(edges):
            graph = defaultdict(list)
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)
            return graph
        
        def bfs(start, graph):
            queue = deque([start])
            visited = {start}
            farthest_node = start
            farthest_distance = 0
            while queue:
                for _ in range(len(queue)):
                    node = queue.popleft()
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                            if len(visited) > farthest_distance:
                                farthest_node = neighbor
                                farthest_distance = len(visited)
            return farthest_node, farthest_distance - 1
        
        def get_diameter(graph):
            start_node, _ = bfs(0, graph)
            _, diameter = bfs(start_node, graph)
            return diameter
        
        graph1 = build_graph(edges1)
        graph2 = build_graph(edges2)
        
        diameter1 = get_diameter(graph1)
        diameter2 = get_diameter(graph2)
        
        min_diameter = min(diameter1 + 1, diameter2 + 1)
        
        return min_diameter