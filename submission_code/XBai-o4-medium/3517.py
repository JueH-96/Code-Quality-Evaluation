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
            # BFS to find the shortest path from 0 to n-1
            dist = [-1] * n
            q = deque()
            q.append(0)
            dist[0] = 0
            while q:
                node = q.popleft()
                for neighbor in adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        q.append(neighbor)
            answer.append(dist[n-1])
        
        return answer