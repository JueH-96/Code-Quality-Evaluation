from collections import deque
from typing import List

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize the graph
        graph = [[] for _ in range(n)]
        
        # Add the initial unidirectional roads
        for i in range(n-1):
            graph[i].append(i+1)
        
        answer = []
        
        # Process each query
        for u, v in queries:
            graph[u].append(v)
            
            # Find the shortest path using BFS
            queue = deque([0])
            distances = [-1] * n
            distances[0] = 0
            
            while queue and distances[n-1] == -1:
                city = queue.popleft()
                
                for neighbor in graph[city]:
                    if distances[neighbor] == -1:
                        distances[neighbor] = distances[city] + 1
                        queue.append(neighbor)
            
            # Shortest path to city n-1 is stored in distances[n-1]
            answer.append(distances[n-1])
        
        return answer