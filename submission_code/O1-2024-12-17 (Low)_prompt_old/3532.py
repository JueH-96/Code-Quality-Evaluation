class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        from collections import deque
        
        n = len(edges) + 1
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Cost of visiting a neighbor v is c[v] = 1 if v is odd, else 2
        # Distances in this tree are defined by these (1,2) edge costs.
        def cost(v):
            return 1 if (v % 2) == 1 else 2
        
        # We can perform a BFS-like D’ search (often called 0-1 BFS) for 1,2 edge costs
        # using a deque to keep the complexity O(n).
        def bfs_12(start):
            dist = [float('inf')] * n
            dist[start] = 0
            dq = deque([start])
            while dq:
                u = dq.popleft()
                for v in adj[u]:
                    w = cost(v)  # edge cost to go from u -> v
                    nd = dist[u] + w
                    if nd < dist[v]:
                        dist[v] = nd
                        # If cost is 1, we push to front; if cost is 2, push to back
                        if w == 1:
                            dq.appendleft(v)
                        else:
                            dq.append(v)
            return dist
        
        # 1) Arbitrary BFS from node 0 to find farthest node p
        dist0 = bfs_12(0)
        p = max(range(n), key=lambda x: dist0[x])
        
        # 2) BFS from p to find farthest node q, and record distances dist_p
        dist_p = bfs_12(p)
        q = max(range(n), key=lambda x: dist_p[x])
        
        # 3) BFS from q to get dist_q
        dist_q = bfs_12(q)
        
        # For each node x, the time to mark all is the eccentricity:
        # times[x] = max(dist_p[x], dist_q[x])
        # because p and q are diameter endpoints in this 1,2-weighted tree.
        return [max(dist_p[x], dist_q[x]) for x in range(n)]