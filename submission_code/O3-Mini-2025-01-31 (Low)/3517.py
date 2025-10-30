from collections import deque
from typing import List

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Build initial graph: road from city i to city i+1 (for 0 <= i < n-1)
        graph = [[] for _ in range(n)]
        for i in range(n - 1):
            graph[i].append(i + 1)
        
        def bfs() -> int:
            # Run a BFS from city 0 to city n-1.
            dist = [-1] * n
            dist[0] = 0
            dq = deque([0])
            while dq:
                node = dq.popleft()
                if node == n - 1:
                    return dist[node]
                for nei in graph[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        dq.append(nei)
            # if n - 1 not reached, return -1 (though problem guarantees connectivity due to initial roads)
            return -1
        
        res = []
        # Process each query by adding the extra road and compute new shortest path.
        for u, v in queries:
            graph[u].append(v)
            res.append(bfs())
        return res