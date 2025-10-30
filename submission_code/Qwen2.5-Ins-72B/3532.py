from collections import defaultdict, deque
from typing import List

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def bfs(start):
            marked = [False] * n
            marked[start] = True
            time = [0] * n
            queue = deque([start])
            level = 0
            
            while queue:
                for _ in range(len(queue)):
                    node = queue.popleft()
                    for neighbor in graph[node]:
                        if not marked[neighbor]:
                            if (node % 2 == 1 and level % 2 == 1) or (node % 2 == 0 and level % 2 == 0):
                                marked[neighbor] = True
                                time[neighbor] = level + 1
                                queue.append(neighbor)
                            elif (node % 2 == 1 and level % 2 == 0) or (node % 2 == 0 and level % 2 == 1):
                                marked[neighbor] = True
                                time[neighbor] = level + 2
                                queue.append(neighbor)
                level += 1
            
            return max(time)
        
        result = [0] * n
        for i in range(n):
            result[i] = bfs(i)
        
        return result