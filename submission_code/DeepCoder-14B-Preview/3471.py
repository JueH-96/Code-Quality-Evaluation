from collections import deque

class Solution:
    def minimumDiameterAfterMerge(self, edges1: list[list[int]], edges2: list[list[int]]) -> int:
        def compute_diameter_and_eccentricities(edges, size):
            if size == 1:
                return (0, [0])
            
            adj = [[] for _ in range(size)]
            for a, b in edges:
                adj[a].append(b)
                adj[b].append(a)
            
            def bfs(start):
                visited = [False] * size
                q = deque()
                q.append((start, 0))
                visited[start] = True
                max_dist = 0
                far_node = start
                dist = [0] * size
                while q:
                    node, d = q.popleft()
                    dist[node] = d
                    if d > max_dist:
                        max_dist = d
                        far_node = node
                    for neighbor in adj[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            q.append((neighbor, d + 1))
                return far_node, dist
            
            start = 0
            far_node, _ = bfs(start)
            far_node2, dist_from_far = bfs(far_node)
            diam = dist_from_far[far_node2]
            
            dist1 = bfs(far_node)[1]
            dist2 = bfs(far_node2)[1]
            
            ecc = [max(dist1[u], dist2[u]) for u in range(size)]
            return diam, ecc
        
        n = len(edges1) + 1 if edges1 else 1
        m = len(edges2) + 1 if edges2 else 1
        
        if n == 1:
            diam1 = 0
            ecc1 = [0]
        else:
            diam1, ecc1 = compute_diameter_and_eccentricities(edges1, n)
        
        if m == 1:
            diam2 = 0
            ecc2 = [0]
        else:
            diam2, ecc2 = compute_diameter_and_eccentricities(edges2, m)
        
        min_ecc1 = min(ecc1) if ecc1 else 0
        min_ecc2 = min(ecc2) if ecc2 else 0
        
        s_min = min_ecc1 + min_ecc2 + 1
        D = max(diam1, diam2)
        
        minimal_max = max(D, s_min)
        return minimal_max