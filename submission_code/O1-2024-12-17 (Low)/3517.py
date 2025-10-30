class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        from collections import deque
        
        # Build the initial adjacency list for the given n
        # There is a unidirectional road from i -> i+1 for i in [0..n-2]
        adj = [[] for _ in range(n)]
        for i in range(n - 1):
            adj[i].append(i + 1)
        
        # A helper function to compute the shortest distance
        # from city 0 to city n-1 using BFS
        def bfs_shortest_path():
            dist = [-1] * n
            dist[0] = 0
            q = deque([0])
            
            while q:
                node = q.popleft()
                # If we've reached city n-1, we can return the distance
                if node == n - 1:
                    return dist[node]
                
                for nei in adj[node]:
                    if dist[nei] == -1:  # not visited
                        dist[nei] = dist[node] + 1
                        q.append(nei)
            
            # If somehow unreachable (though problem constraints always allow a path),
            # we would return -1 or some indication. By problem statement, this shouldn't happen.
            return -1
        
        answer = []
        
        # Process each query, adding a new road, and compute shortest path
        for u, v in queries:
            adj[u].append(v)
            answer.append(bfs_shortest_path())
        
        return answer