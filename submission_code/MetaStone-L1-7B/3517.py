from collections import deque
from typing import List

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize adjacency list with the default roads
        adj = [[] for _ in range(n)]
        for i in range(n-1):
            adj[i].append(i + 1)
        
        res = []
        for u, v in queries:
            # Add the new road to the adjacency list
            adj[u].append(v)
            
            # BFS to find the shortest path
            distance = [-1] * n
            distance[0] = 0
            q = deque([0])
            while q:
                current = q.popleft()
                if current == n - 1:
                    res.append(distance[current])
                    break  # Early exit as BFS finds the shortest path
                for neighbor in adj[current]:
                    if distance[neighbor] == -1:
                        distance[neighbor] = distance[current] + 1
                        q.append(neighbor)
        return res