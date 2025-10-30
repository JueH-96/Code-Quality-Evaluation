from collections import deque
from typing import List

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize adjacency list with the original roads
        adj = [[] for _ in range(n)]
        for i in range(n - 1):
            adj[i].append(i + 1)
        
        # Initialize the distance array with the original path lengths
        distance = list(range(n))
        answer = []
        
        for u, v in queries:
            # Add the new road to the adjacency list
            adj[u].append(v)
            
            # Check if the new road provides a shorter path to v
            if distance[u] + 1 < distance[v]:
                distance[v] = distance[u] + 1
                # Use BFS to propagate the updated distance to subsequent cities
                queue = deque([v])
                while queue:
                    current = queue.popleft()
                    for neighbor in adj[current]:
                        if distance[neighbor] > distance[current] + 1:
                            distance[neighbor] = distance[current] + 1
                            queue.append(neighbor)
            
            # Record the current shortest distance to the last city
            answer.append(distance[-1])
        
        return answer