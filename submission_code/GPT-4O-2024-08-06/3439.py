from typing import List
from collections import deque

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        
        def tree_diameter(edges: List[List[int]], n: int) -> int:
            # Build adjacency list
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Function to perform BFS and return the farthest node and its distance
            def bfs(start: int) -> (int, int):
                queue = deque([start])
                visited = [-1] * n
                visited[start] = 0
                farthest_node = start
                max_distance = 0
                
                while queue:
                    node = queue.popleft()
                    for neighbor in adj[node]:
                        if visited[neighbor] == -1:
                            visited[neighbor] = visited[node] + 1
                            queue.append(neighbor)
                            if visited[neighbor] > max_distance:
                                max_distance = visited[neighbor]
                                farthest_node = neighbor
                return farthest_node, max_distance
            
            # Find the farthest node from any node (say node 0)
            farthest_node, _ = bfs(0)
            # Find the farthest node from the farthest node found above
            _, diameter = bfs(farthest_node)
            
            return diameter
        
        # Calculate the diameters of both trees
        n1 = len(edges1) + 1
        n2 = len(edges2) + 1
        diameter1 = tree_diameter(edges1, n1)
        diameter2 = tree_diameter(edges2, n2)
        
        # The minimum possible diameter after connecting the two trees
        # is the maximum of the two diameters plus 1
        return max(diameter1, diameter2) + 1