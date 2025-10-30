from collections import deque
from typing import List

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        # Helper function to compute diameter of a tree given its edge list
        def tree_diameter(edges: List[List[int]]) -> int:
            if not edges:
                return 0
                
            # Build graph. Number of nodes = len(edges) + 1.
            n = len(edges) + 1
            graph = [[] for _ in range(n)]
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)
            
            # BFS to find the farthest node from an arbitrary starting node (say 0)
            def bfs(start: int) -> (int, int):
                visited = [False] * n
                dist = [0] * n
                queue = deque([start])
                visited[start] = True
                while queue:
                    cur = queue.popleft()
                    for nei in graph[cur]:
                        if not visited[nei]:
                            visited[nei] = True
                            dist[nei] = dist[cur] + 1
                            queue.append(nei)
                # find the farthest node and its distance
                farthest = 0
                max_dist = 0
                for i in range(n):
                    if dist[i] > max_dist:
                        max_dist = dist[i]
                        farthest = i
                return farthest, max_dist
            
            # First BFS from node 0 (or any) to find one end of the diameter
            far_node, _ = bfs(0)
            # Second BFS from that farthest node to get the diameter length
            _, diameter = bfs(far_node)
            return diameter
        
        # Compute diameters for both trees
        d1 = tree_diameter(edges1)
        d2 = tree_diameter(edges2)
        
        # The best result after merging will be the maximum of:
        #   - the two diameters, and
        #   - the sum of the radii of the trees plus one (for the new connecting edge)
        # Radius is defined as (diameter + 1) // 2.
        r1 = (d1 + 1) // 2
        r2 = (d2 + 1) // 2
        
        return max(d1, d2, r1 + r2 + 1)