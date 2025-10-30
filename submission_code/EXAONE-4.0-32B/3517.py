from collections import deque
from typing import List

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        graph = [[] for _ in range(n)]
        for i in range(n-1):
            graph[i].append(i+1)
        
        result = []
        for u, v in queries:
            graph[u].append(v)
            dist = [-1] * n
            dist[0] = 0
            queue = deque([0])
            while queue:
                current = queue.popleft()
                if current == n-1:
                    result.append(dist[current])
                    break
                for neighbor in graph[current]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[current] + 1
                        queue.append(neighbor)
        return result