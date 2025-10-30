from collections import deque
from typing import List

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        adj = [[] for _ in range(n)]
        for i in range(n - 1):
            adj[i].append(i + 1)
        
        ans = []
        for query in queries:
            u, v = query
            adj[u].append(v)
            
            dist = [-1] * n
            q = deque([0])
            dist[0] = 0
            
            while q:
                curr = q.popleft()
                for neighbor in adj[curr]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[curr] + 1
                        q.append(neighbor)
            
            ans.append(dist[n - 1])
        
        return ans