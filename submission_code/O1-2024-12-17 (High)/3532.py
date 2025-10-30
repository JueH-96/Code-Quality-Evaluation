class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        from collections import deque

        n = len(edges) + 1
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # c[i] = cost(node i) = 2 if i is even, 1 if i is odd
        c = [2 if i % 2 == 0 else 1 for i in range(n)]
        
        def bfs(start: int) -> Tuple[int, List[int]]:
            """
            Returns (farthest_node, dist[]) starting from 'start',
            where dist[v] = sum of c(...) along the path from 'start' to v,
            including c(start) and c(v).
            """
            dist = [-1] * n
            dist[start] = c[start]
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                for w in adj[u]:
                    if dist[w] < 0:  # not visited
                        dist[w] = dist[u] + c[w]
                        queue.append(w)
            
            # Find the node with the maximum distance
            far_node = 0
            max_d = dist[0]
            for i in range(n):
                if dist[i] > max_d:
                    max_d = dist[i]
                    far_node = i
            return far_node, dist
        
        # 1) Arbitrary start from node 0 to find farthest node Y
        Y, _ = bfs(0)
        # 2) BFS from Y to find farthest node Z and record distY
        Z, distY = bfs(Y)
        # 3) BFS from Z to get distZ
        _, distZ = bfs(Z)
        
        # times[x] = max(distY[x], distZ[x]) - c[x]
        # because the path-sum includes c(x) in distY[x] or distZ[x],
        # but the problem's marking-time excludes the cost of the starting node itself.
        times = [max(distY[i], distZ[i]) - c[i] for i in range(n)]
        
        return times