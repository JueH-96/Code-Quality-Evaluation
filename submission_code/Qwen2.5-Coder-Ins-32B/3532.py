from typing import List
from collections import defaultdict, deque

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        graph = defaultdict(list)
        
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def bfs(start):
            queue = deque([(start, 0)])
            visited = [False] * n
            visited[start] = True
            times = [0] * n
            
            while queue:
                node, time = queue.popleft()
                times[node] = time
                
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        if node % 2 == 0:
                            queue.append((neighbor, time + 2))
                        else:
                            queue.append((neighbor, time + 1))
            
            return max(times)
        
        result = [0] * n
        for i in range(n):
            result[i] = bfs(i)
        
        return result