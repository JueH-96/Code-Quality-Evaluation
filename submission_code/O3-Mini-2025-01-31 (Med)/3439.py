from collections import defaultdict, deque
from typing import List

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        
        def tree_diameter(n: int, edges: List[List[int]]) -> int:
            # Build the adjacency list for the tree.
            graph = defaultdict(list)
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)
            
            # Helper function: perform BFS from start and return (farthest_node, distance list)
            def bfs(start: int):
                dist = [-1] * n
                q = deque([start])
                dist[start] = 0
                while q:
                    node = q.popleft()
                    for nei in graph[node]:
                        if dist[nei] == -1:
                            dist[nei] = dist[node] + 1
                            q.append(nei)
                # Find the furthest node from 'start'
                farthest_node = 0
                for i in range(n):
                    if dist[i] > dist[farthest_node]:
                        farthest_node = i
                return farthest_node, dist
            
            # Perform the two BFS's to get the diameter of the tree.
            node1, _ = bfs(0)          # BFS from an arbitrary node (node 0)
            node2, dist = bfs(node1)    # BFS from the farthest node found
            diameter = dist[node2]
            return diameter
        
        # Number of nodes in each tree can be deduced as number of edges + 1.
        n = len(edges1) + 1
        m = len(edges2) + 1
        
        # Calculate diameters for both trees.
        d1 = tree_diameter(n, edges1)
        d2 = tree_diameter(m, edges2)
        
        # The "radius" (or half-length rounded up) in each tree.
        h1 = (d1 + 1) // 2
        h2 = (d2 + 1) // 2
        
        # When the two trees are connected optimally (by linking near their centers), 
        # the resulting diameter is the maximum among the two individual diameters 
        # and the sum of the two radii plus one (for the connecting edge).
        return max(d1, d2, h1 + h2 + 1)