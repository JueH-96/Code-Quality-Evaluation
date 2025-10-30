from typing import List
from collections import deque

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize adjacency list with the initial chain of roads 0->1->2->...->n-1
        adj = [[] for _ in range(n)]
        for i in range(n - 1):
            adj[i].append(i + 1)
        
        answer = []
        for u, v in queries:
            # Add the new road
            adj[u].append(v)
            
            # BFS to find shortest path from 0 to n-1
            dist = [-1] * n
            dist[0] = 0
            dq = deque([0])
            while dq:
                x = dq.popleft()
                if x == n - 1:
                    break
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        dq.append(y)
            
            answer.append(dist[n - 1])
        
        return answer