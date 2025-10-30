from collections import deque
from typing import List

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize adjacency list with the initial roads
        adj = [[] for _ in range(n)]
        for i in range(n - 1):
            adj[i].append(i + 1)
        
        answer = []
        for u, v in queries:
            # Add the new edge to the adjacency list
            adj[u].append(v)
            # BFS to find the shortest distance from 0 to n-1
            dist = [-1] * n
            dist[0] = 0
            q = deque([0])
            found = False
            while q:
                current = q.popleft()
                if current == n - 1:
                    found = True
                    break
                for neighbor in adj[current]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[current] + 1
                        q.append(neighbor)
            answer.append(dist[n - 1])
        return answer