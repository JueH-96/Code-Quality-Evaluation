from collections import deque, defaultdict
from typing import List

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        def bfs(start):
            times = [-1] * n
            times[start] = 0
            queue = deque([start])
            
            while queue:
                node = queue.popleft()
                current_time = times[node]
                
                for neighbor in graph[node]:
                    if times[neighbor] == -1:
                        if neighbor % 2 == 0:
                            times[neighbor] = current_time + 2
                        else:
                            times[neighbor] = current_time + 1
                        queue.append(neighbor)
            
            return max(times)
        
        n = len(edges) + 1
        graph = defaultdict(list)
        
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        result = []
        for i in range(n):
            result.append(bfs(i))
        
        return result