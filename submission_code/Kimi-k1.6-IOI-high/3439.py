from collections import deque
from typing import List

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def get_diameter(edges):
            if not edges:
                return 0
            adj = {}
            for a, b in edges:
                if a not in adj:
                    adj[a] = []
                if b not in adj:
                    adj[b] = []
                adj[a].append(b)
                adj[b].append(a)
            
            def bfs(start_node):
                visited = {}
                q = deque()
                q.append((start_node, 0))
                visited[start_node] = 0
                max_dist = -1
                far_node = start_node
                while q:
                    node, dist = q.popleft()
                    if dist > max_dist:
                        max_dist = dist
                        far_node = node
                    for nei in adj.get(node, []):
                        if nei not in visited:
                            visited[nei] = dist + 1
                            q.append((nei, dist + 1))
                return far_node, max_dist
            
            # Handle single node case (possible if edges is empty but node exists, but edges is empty implies single node)
            start = next(iter(adj.keys())) if adj else 0
            u, _ = bfs(start)
            v, diameter = bfs(u)
            return diameter
        
        d1 = get_diameter(edges1)
        d2 = get_diameter(edges2)
        original_max = max(d1, d2)
        radius1 = (d1 + 1) // 2
        radius2 = (d2 + 1) // 2
        sum_min = radius1 + radius2 + 1
        return max(original_max, sum_min)