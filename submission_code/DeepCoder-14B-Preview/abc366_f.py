from collections import deque
from typing import List

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1  # since edges has n-1 elements
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        result = [0] * n
        
        for i in range(n):
            times = [-1] * n
            times[i] = 0
            q = deque()
            q.append(i)
            
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if times[v] == -1:
                        if v % 2 == 0:
                            times[v] = times[u] + 2
                        else:
                            times[v] = times[u] + 1
                        q.append(v)
            
            max_time = max(times)
            result[i] = max_time
        
        return result