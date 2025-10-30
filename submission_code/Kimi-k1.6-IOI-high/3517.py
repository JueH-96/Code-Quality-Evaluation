from typing import List
from collections import deque

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        adj = [[] for _ in range(n)]
        for i in range(n - 1):
            adj[i].append(i + 1)
        answer = []
        
        for u, v in queries:
            adj[u].append(v)
            dist = [float('inf')] * n
            dist[0] = 0
            queue = deque([0])
            
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if dist[neighbor] > dist[node] + 1:
                        dist[neighbor] = dist[node] + 1
                        queue.append(neighbor)
            
            answer.append(dist[n - 1])
        
        return answer