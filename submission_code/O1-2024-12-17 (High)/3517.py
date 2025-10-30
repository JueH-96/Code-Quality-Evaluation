class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        from collections import deque
        
        # Build the initial graph with roads from i -> i+1
        adjacency = [[] for _ in range(n)]
        for i in range(n - 1):
            adjacency[i].append(i + 1)
        
        def bfs_shortest_path() -> int:
            """Return the shortest distance from city 0 to city n-1 using BFS."""
            dist = [-1] * n
            dist[0] = 0
            queue = deque([0])
            
            while queue:
                city = queue.popleft()
                if city == n - 1:
                    return dist[city]
                for nxt in adjacency[city]:
                    if dist[nxt] == -1:
                        dist[nxt] = dist[city] + 1
                        queue.append(nxt)
            
            # If not reachable (the problem constraints imply this won't happen),
            # we could return -1 or some sentinel. We'll use -1 here.
            return -1
        
        result = []
        # Process each query and compute the shortest path after adding the new road
        for u, v in queries:
            adjacency[u].append(v)
            result.append(bfs_shortest_path())
        
        return result