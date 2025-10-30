from collections import deque
from typing import List

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        w = [1 if i % 2 == 1 else 2 for i in range(n)]
        
        def bfs(start):
            dist = [-1] * n
            q = deque()
            dist[start] = w[start]
            q.append(start)
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + w[v]
                        q.append(v)
            return dist
        
        dist0 = bfs(0)
        A = dist0.index(max(dist0))
        distA = bfs(A)
        B = distA.index(max(distA))
        distB = bfs(B)
        
        ans = [0] * n
        for i in range(n):
            ecc = max(distA[i], distB[i])
            ans[i] = ecc - w[i]
        return ans