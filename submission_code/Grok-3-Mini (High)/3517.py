from typing import List
from collections import deque

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize adjacency list
        adj = [[] for _ in range(n)]
        # Add initial unidirectional roads from i to i+1
        for i in range(n - 1):
            adj[i].append(i + 1)
        
        answer = []
        
        # Process each query
        for query in queries:
            u, v = query[0], query[1]
            # Add the new unidirectional road from u to v
            adj[u].append(v)
            
            # BFS to find the shortest path from city 0 to city n-1
            dist = [-1] * n  # -1 indicates unvisited
            dist[0] = 0
            queue = deque([0])
            
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if dist[neighbor] == -1:  # If not visited
                        dist[neighbor] = dist[node] + 1
                        queue.append(neighbor)
            
            # Append the distance to city n-1
            answer.append(dist[n - 1])
        
        return answer