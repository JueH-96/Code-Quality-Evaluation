from typing import List
from collections import deque

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize the graph with the initial roads
        graph = [[] for _ in range(n)]
        for i in range(n - 1):
            graph[i].append(i + 1)
        
        answer = []
        
        for query in queries:
            u, v = query
            graph[u].append(v)
            dist = self.bfs(graph, 0, n - 1)
            answer.append(dist)
        
        return answer
    
    def bfs(self, graph, start, end):
        n = len(graph)
        dist = [float('inf')] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if u == end:
                return dist[end]
            for v in graph[u]:
                if dist[u] + 1 < dist[v]:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        
        return -1  # Should not happen as end is always reachable