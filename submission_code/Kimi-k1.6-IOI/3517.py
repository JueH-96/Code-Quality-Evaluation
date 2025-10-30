from collections import deque
from typing import List

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize the adjacency list with the original roads
        adj = [[] for _ in range(n)]
        for i in range(n - 1):
            adj[i].append(i + 1)
        
        result = []
        
        for u, v in queries:
            adj[u].append(v)
            # Perform BFS to find the shortest path from 0 to n-1
            distance = [float('inf')] * n
            distance[0] = 0
            queue = deque([0])
            
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if distance[neighbor] == float('inf'):
                        distance[neighbor] = distance[node] + 1
                        queue.append(neighbor)
            
            result.append(distance[-1])
        
        return result