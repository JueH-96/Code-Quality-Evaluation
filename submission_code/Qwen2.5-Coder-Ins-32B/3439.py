from typing import List
from collections import defaultdict, deque

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def bfs(graph, start):
            visited = [False] * len(graph)
            queue = deque([(start, 0)])
            visited[start] = True
            farthest_node = start
            max_distance = 0
            
            while queue:
                node, distance = queue.popleft()
                if distance > max_distance:
                    max_distance = distance
                    farthest_node = node
                
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append((neighbor, distance + 1))
            
            return farthest_node, max_distance
        
        def diameter(graph):
            farthest_node, _ = bfs(graph, 0)
            _, diameter = bfs(graph, farthest_node)
            return diameter
        
        def build_graph(edges, n):
            graph = defaultdict(list)
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)
            return graph
        
        n = len(edges1) + 1
        m = len(edges2) + 1
        
        graph1 = build_graph(edges1, n)
        graph2 = build_graph(edges2, m)
        
        diameter1 = diameter(graph1)
        diameter2 = diameter(graph2)
        
        max_diameter = max(diameter1, diameter2)
        
        for u in range(n):
            for v in range(m):
                # Simulate connecting u from graph1 to v from graph2
                # Calculate the new diameter
                # We need to find the longest path that includes the new edge (u, v)
                # This can be done by finding the farthest node from u in graph1 and the farthest node from v in graph2
                farthest_in_graph1, dist1 = bfs(graph1, u)
                farthest_in_graph2, dist2 = bfs(graph2, v)
                
                # The new diameter is the maximum of the old diameters and the path through the new edge
                new_diameter = max(diameter1, diameter2, dist1 + dist2 + 1)
                max_diameter = min(max_diameter, new_diameter)
        
        return max_diameter