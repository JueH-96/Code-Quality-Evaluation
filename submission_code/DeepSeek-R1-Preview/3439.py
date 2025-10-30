from typing import List
from collections import defaultdict, deque

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        # Compute diameters of both trees
        d1 = self.compute_diameter(edges1)
        d2 = self.compute_diameter(edges2)
        
        # Compute radii
        r1 = (d1 + 1) // 2
        r2 = (d2 + 1) // 2
        
        # The new possible diameter
        candidate = r1 + r2 + 1
        new_diameter = max(d1, d2, candidate)
        
        return new_diameter
    
    def compute_diameter(self, edges: List[List[int]]) -> int:
        if not edges:
            return 0
        
        # Build adjacency list
        adj = defaultdict(list)
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        
        # Find all nodes present in the tree
        nodes = set()
        for a, b in edges:
            nodes.add(a)
            nodes.add(b)
        start_node = next(iter(nodes))  # pick any node as the starting point
        
        def bfs(start):
            q = deque()
            q.append((start, -1, 0))  # (node, parent, distance)
            max_dist = 0
            far_node = start
            while q:
                node, parent, dist = q.popleft()
                if dist > max_dist:
                    max_dist = dist
                    far_node = node
                for neighbor in adj[node]:
                    if neighbor != parent:
                        q.append((neighbor, node, dist + 1))
            return far_node, max_dist
        
        # First BFS to find the farthest node from the start node
        u, _ = bfs(start_node)
        # Second BFS to find the diameter
        v, diameter = bfs(u)
        return diameter