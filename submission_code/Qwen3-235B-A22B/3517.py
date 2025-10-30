from collections import deque
from typing import List

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize adjacency list with the initial roads
        adj = [[] for _ in range(n)]
        for i in range(n - 1):
            adj[i].append(i + 1)
        
        result = []
        
        for u, v in queries:
            # Add the new road to the adjacency list
            adj[u].append(v)
            
            # Perform BFS to find the shortest path from 0 to n-1
            visited = [False] * n
            distance = [0] * n
            q = deque()
            q.append(0)
            visited[0] = True
            found = False
            
            while q and not found:
                current = q.popleft()
                for neighbor in adj[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        distance[neighbor] = distance[current] + 1
                        if neighbor == n - 1:
                            found = True
                            result.append(distance[neighbor])
                            break
                        q.append(neighbor)
                if found:
                    break
            
            # If not found (theoretically impossible here), append the distance to n-1
            if not found:
                result.append(distance[n - 1])
        
        return result