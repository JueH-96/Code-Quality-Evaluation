from typing import List
from collections import defaultdict, deque

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
            distance = {start: 0}
            farthest_node = start
            
            while queue:
                node = queue.popleft()
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distance[neighbor] = distance[node] + 1
                        queue.append(neighbor)
                        if distance[neighbor] > distance[farthest_node]:
                            farthest_node = neighbor
            
            return farthest_node, distance
        
        # Build the graphs for both trees
        graph1 = build_graph(edges1)
        graph2 = build_graph(edges2)
        
        # Find the diameter of the first tree
        farthest_node1, _ = bfs(0, graph1)
        farthest_node1, dist1 = bfs(farthest_node1, graph1)
        diameter1 = max(dist1.values())
        
        # Find the diameter of the second tree
        farthest_node2, _ = bfs(0, graph2)
        farthest_node2, dist2 = bfs(farthest_node2, graph2)
        diameter2 = max(dist2.values())
        
        # The minimum diameter after merging
        return diameter1 + diameter2 + 1