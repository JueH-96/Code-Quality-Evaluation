from typing import List
from collections import deque

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize the adjacency list with the initial roads
        adj = [[] for _ in range(n)]
        for i in range(n-1):
            adj[i].append(i+1)
        
        res = []
        for u, v in queries:
            # Add the new edge from u to v
            adj[u].append(v)
            
            # BFS to find the shortest path from 0 to n-1
            visited = [False] * n
            dist = [float('inf')] * n
            q = deque()
            q.append(0)
            dist[0] = 0
            visited[0] = True
            found = False
            while q:
                node = q.popleft()
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        dist[neighbor] = dist[node] + 1
                        q.append(neighbor)
                        if neighbor == n-1:
                            # Early termination since we found the target
                            found = True
                            break
                if found:
                    break
            res.append(dist[n-1])
        
        return res