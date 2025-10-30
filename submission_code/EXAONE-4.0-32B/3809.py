from collections import deque
from typing import List

class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)
        sets_list = [set(row) for row in properties]
        
        graph = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                common = sets_list[i] & sets_list[j]
                if len(common) >= k:
                    graph[i].append(j)
                    graph[j].append(i)
        
        visited = [False] * n
        components = 0
        
        for i in range(n):
            if not visited[i]:
                components += 1
                queue = deque([i])
                visited[i] = True
                while queue:
                    node = queue.popleft()
                    for neighbor in graph[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
        
        return components