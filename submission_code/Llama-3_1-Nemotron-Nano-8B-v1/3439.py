from typing import List
from collections import deque

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def process_tree(edges, n):
            if n == 1:
                return 0, 0
            
            adj = [[] for _ in range(n)]
            for a, b in edges:
                adj[a].append(b)
                adj[b].append(a)
            
            # Find the farthest node from an arbitrary node (0)
            u = self.bfs_farthest(0, adj)[0]
            
            # Find the diameter endpoints (u and v)
            v, d = self.bfs_farthest(u, adj)
            
            # Compute distances from u and v
            dist_u = self.bfs_distance(u, adj)
            dist_v = self.bfs_distance(v, adj)
            
            # Find the minimal radius
            min_r = min(max(dist_u[i], dist_v[i]) for i in range(n))
            return d, min_r
        
        def bfs_farthest(start, adj):
            visited = [False] * len(adj)
            queue = deque([(start, 0)])
            visited[start] = True
            max_dist = 0
            farthest_node = start
            while queue:
                node, dist = queue.popleft()
                if dist > max_dist:
                    max_dist = dist
                    farthest_node = node
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append((neighbor, dist + 1))
            return farthest_node, max_dist
        
        def bfs_distance(start, adj):
            n = len(adj)
            dist = [-1] * n
            queue = deque([start])
            dist[start] = 0
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        queue.append(neighbor)
            return dist
        
        # Process the first tree
        d1, min_r1 = process_tree(edges1, len(edges1) + 1)
        # Process the second tree
        d2, min_r2 = process_tree(edges2, len(edges2) + 1)
        
        M = max(d1, d2)
        sum_min = min_r1 + min_r2
        answer = max(M, sum_min + 1)
        
        return answer