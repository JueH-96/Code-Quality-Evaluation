from typing import List
from collections import deque

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize adjacency list with the initial roads
        adj = [[] for _ in range(n)]
        for i in range(n - 1):
            adj[i].append(i + 1)
        
        answer = []
        
        for u, v in queries:
            adj[u].append(v)  # Add the new edge from the current query
            
            # Perform BFS to find the shortest path from 0 to n-1
            dist = [float('inf')] * n
            dist[0] = 0
            q = deque([0])
            
            while q:
                node = q.popleft()
                # Early termination if we reach the target node
                if node == n - 1:
                    break
                for neighbor in adj[node]:
                    if dist[neighbor] > dist[node] + 1:
                        dist[neighbor] = dist[node] + 1
                        q.append(neighbor)
            
            answer.append(dist[n - 1])
        
        return answer