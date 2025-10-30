class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        from collections import deque

        def get_diameter(n, edges):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # First BFS to find one end of the diameter
            dist = [-1] * n
            q = deque()
            q.append(0)
            dist[0] = 0
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            u1 = dist.index(max(dist))
            
            # Second BFS from farthest node found
            dist = [-1] * n
            q = deque()
            q.append(u1)
            dist[u1] = 0
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            D = max(dist)
            return D

        n1 = max(max(u, v) for u, v in edges1) + 1 if edges1 else 1
        n2 = max(max(u, v) for u, v in edges2) + 1 if edges2 else 1

        D1 = get_diameter(n1, edges1)
        D2 = get_diameter(n2, edges2)

        diam = max((D1 + 1) // 2 + (D2 + 1) // 2 + 1, D1, D2)
        return diam