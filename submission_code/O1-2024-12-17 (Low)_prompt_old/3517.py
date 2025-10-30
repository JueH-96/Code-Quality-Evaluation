class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        from collections import deque
        
        # Build initial adjacency list with original unidirectional roads
        adjacency = [[] for _ in range(n)]
        for i in range(n - 1):
            adjacency[i].append(i + 1)
        
        answer = []

        def bfs_shortest_path():
            # Returns the shortest distance from city 0 to city n-1 using BFS
            # or -1 if no path exists (though the problem examples do not show no-path cases).
            visited = [False] * n
            dist = [float('inf')] * n
            queue = deque([0])
            visited[0] = True
            dist[0] = 0
            
            while queue:
                cur = queue.popleft()
                if cur == n - 1:
                    return dist[cur]  # Found the shortest path to n-1
                
                for nxt in adjacency[cur]:
                    if not visited[nxt]:
                        visited[nxt] = True
                        dist[nxt] = dist[cur] + 1
                        queue.append(nxt)
            
            return -1  # In case no path is found
        
        # Process each query by adding the new road and then compute BFS
        for u, v in queries:
            adjacency[u].append(v)
            answer.append(bfs_shortest_path())
        
        return answer