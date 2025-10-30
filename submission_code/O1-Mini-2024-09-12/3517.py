from collections import deque
from typing import List

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        adjacency = [[] for _ in range(n)]
        # Initialize with roads from i to i+1
        for i in range(n - 1):
            adjacency[i].append(i + 1)
        
        answers = []
        
        for u, v in queries:
            adjacency[u].append(v)
            # BFS from 0
            queue = deque()
            queue.append(0)
            visited = [False] * n
            visited[0] = True
            distance = 0
            found = False
            while queue and not found:
                for _ in range(len(queue)):
                    current = queue.popleft()
                    if current == n - 1:
                        found = True
                        break
                    for neighbor in adjacency[current]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                if not found:
                    distance += 1
            answers.append(distance if found else -1)
        
        return answers