from typing import List
from collections import deque

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize the adjacency list with the initial roads
        adj = [[] for _ in range(n)]
        for i in range(n - 1):
            adj[i].append(i + 1)
        
        result = []
        for u, v in queries:
            # Add the new road from u to v
            adj[u].append(v)
            # Compute the shortest path using BFS
            shortest = self.bfs(adj, n)
            result.append(shortest)
        
        return result
    
    def bfs(self, adj: List[List[int]], n: int) -> int:
        visited = [False] * n
        q = deque([0])
        visited[0] = True
        distance = 0
        
        while q:
            level_size = len(q)
            for _ in range(level_size):
                node = q.popleft()
                if node == n - 1:
                    return distance
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        q.append(neighbor)
            distance += 1
        
        return distance  # This line is theoretically unreachable as per problem constraints