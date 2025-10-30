class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        from collections import deque
        
        def build_adj_list(edges):
            n = len(edges) + 1
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            return adj
        
        def bfs_farthest(adj, start):
            n = len(adj)
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            farthest = start
            max_dist = 0
            
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                        if dist[v] > max_dist:
                            max_dist = dist[v]
                            farthest = v
            
            return farthest, max_dist
        
        def find_diameter(edges):
            adj = build_adj_list(edges)
            # Find one end of the diameter
            end1, _ = bfs_farthest(adj, 0)
            # Find the other end and the diameter
            end2, diameter = bfs_farthest(adj, end1)
            return diameter
        
        d1 = find_diameter(edges1)
        d2 = find_diameter(edges2)
        
        # Calculate minimum diameter after merge
        # ceil(d/2) = (d + 1) // 2
        return max(d1, d2, (d1 + 1) // 2 + 1 + (d2 + 1) // 2)