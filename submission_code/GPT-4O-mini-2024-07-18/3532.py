from collections import defaultdict, deque
from typing import List

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        graph = defaultdict(list)
        
        # Build the graph
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def bfs(start: int) -> int:
            queue = deque([start])
            visited = [False] * n
            visited[start] = True
            time = 0
            level = 0
            
            while queue:
                level_size = len(queue)
                time += 1
                for _ in range(level_size):
                    node = queue.popleft()
                    for neighbor in graph[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                if level % 2 == 0:  # even level
                    time += 1
                level += 1
            
            return time - 1  # subtract the last increment which is not needed
        
        result = []
        for i in range(n):
            result.append(bfs(i))
        
        return result