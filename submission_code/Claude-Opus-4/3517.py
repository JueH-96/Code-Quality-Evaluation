class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        from collections import deque
        
        # Initialize adjacency list with initial roads
        graph = [[] for _ in range(n)]
        for i in range(n - 1):
            graph[i].append(i + 1)
        
        def bfs():
            # Find shortest path from 0 to n-1 using BFS
            queue = deque([0])
            visited = [False] * n
            visited[0] = True
            distance = 0
            
            while queue:
                size = len(queue)
                for _ in range(size):
                    curr = queue.popleft()
                    if curr == n - 1:
                        return distance
                    
                    for neighbor in graph[curr]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                
                distance += 1
            
            return -1  # Should not reach here given the problem constraints
        
        result = []
        
        # Process each query
        for u, v in queries:
            # Add new road
            graph[u].append(v)
            
            # Find shortest path after adding this road
            shortest = bfs()
            result.append(shortest)
        
        return result