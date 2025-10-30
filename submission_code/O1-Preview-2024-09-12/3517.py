class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        from collections import deque, defaultdict
        
        # Initialize graph with initial unidirectional edges from i to i+1
        adj = [[] for _ in range(n)]
        for u in range(n - 1):
            adj[u].append(u + 1)
        
        # Initialize distances from 0 to all nodes
        dist = [i for i in range(n)]  # dist[u] = distance from 0 to u
        
        answer = []
        
        # Process each query
        for u, v in queries:
            # Add the new edge to the adjacency list
            adj[u].append(v)
            
            # If the new edge leads to a shorter path to v, update distances
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                # Use a queue to perform BFS and propagate updates
                queue = deque([v])
                while queue:
                    curr = queue.popleft()
                    for nei in adj[curr]:
                        if dist[nei] > dist[curr] + 1:
                            dist[nei] = dist[curr] + 1
                            queue.append(nei)
            # Append the current shortest distance from 0 to n - 1
            answer.append(dist[n - 1])
        
        return answer