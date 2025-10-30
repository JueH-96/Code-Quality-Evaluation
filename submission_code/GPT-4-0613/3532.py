from typing import List
from collections import defaultdict, deque

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        times = [0] * n
        for i in range(n):
            visited = [0] * n
            visited[i] = 1
            queue = deque([(i, 0)])
            while queue:
                node, time = queue.popleft()
                times[i] = max(times[i], time)
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = 1
                        if neighbor % 2 == 0:
                            queue.append((neighbor, time + 2))
                        else:
                            queue.append((neighbor, time + 1))
        return times