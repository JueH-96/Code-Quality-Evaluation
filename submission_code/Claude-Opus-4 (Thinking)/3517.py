class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        from collections import deque
        
        # Build initial graph with roads i->i+1
        graph = [[] for _ in range(n)]
        for i in range(n-1):
            graph[i].append(i+1)
        
        result = []
        
        # Process each query
        for u, v in queries:
            # Add new road
            graph[u].append(v)
            
            # BFS to find shortest path from 0 to n-1
            queue = deque([0])
            visited = [False] * n
            visited[0] = True
            distance = [0] * n
            
            while queue:
                curr = queue.popleft()
                
                if curr == n-1:
                    break
                
                for neighbor in graph[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        distance[neighbor] = distance[curr] + 1
                        queue.append(neighbor)
            
            result.append(distance[n-1])
        
        return result