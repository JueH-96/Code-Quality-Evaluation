from typing import List
from collections import deque

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize adjacency list
        adj = [[] for _ in range(n)]
        for i in range(n - 1):
            adj[i].append(i + 1)
        
        # Initialize distances: initially, distance from 0 to i is i
        dist = list(range(n))
        
        answer = []
        
        for u, v in queries:
            adj[u].append(v)
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                queue = deque([v])
                while queue:
                    w = queue.popleft()
                    for x in adj[w]:
                        if dist[x] > dist[w] + 1:
                            dist[x] = dist[w] + 1
                            queue.append(x)
            answer.append(dist[n - 1])
        
        return answer