from collections import deque

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        answer = []
        
        for u, v in queries:
            graph = [[] for _ in range(n)]
            
            # Initialize the graph with the initial roads
            for i in range(n-1):
                graph[i].append(i+1)
            
            # Add the new road from u to v
            graph[u].append(v)
            
            # Perform BFS to find the shortest path from 0 to n-1
            queue = deque([(0, 0)])
            visited = [False] * n
            visited[0] = True
            
            while queue:
                node, dist = queue.popleft()
                if node == n-1:
                    answer.append(dist)
                    break
                
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append((neighbor, dist+1))
        
        return answer