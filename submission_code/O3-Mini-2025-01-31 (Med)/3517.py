from collections import deque
from typing import List

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize graph with initial roads: i -> i+1 for i in [0, n-2]
        graph = [[] for _ in range(n)]
        for i in range(n - 1):
            graph[i].append(i + 1)
        
        def bfs() -> int:
            # BFS from city 0 to city n - 1
            dist = [-1] * n
            dist[0] = 0
            q = deque([0])
            while q:
                cur = q.popleft()
                if cur == n - 1:
                    return dist[cur]
                for nxt in graph[cur]:
                    if dist[nxt] == -1:
                        dist[nxt] = dist[cur] + 1
                        q.append(nxt)
            return dist[n - 1]  # Should not be -1 because there's always a path
            
        result = []
        # Process each query and add the lead
        for u, v in queries:
            graph[u].append(v)
            # After adding new road, run BFS to get the shortest path length from 0 to n-1
            result.append(bfs())
        return result