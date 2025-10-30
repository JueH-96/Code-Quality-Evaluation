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
        visited = [False] * n
        queue = deque([(0, 0)])
        while queue:
            node, time = queue.popleft()
            visited[node] = True
            times[node] = time
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    if node % 2 == 0 and time % 2 == 0:
                        queue.append((neighbor, time + 2))
                    elif node % 2 == 1 and time % 2 == 1:
                        queue.append((neighbor, time + 2))
                    else:
                        queue.append((neighbor, time + 1))
        return times