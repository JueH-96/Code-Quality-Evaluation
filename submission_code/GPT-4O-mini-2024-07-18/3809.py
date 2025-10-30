from typing import List
from collections import defaultdict

class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        
        def intersect(a, b):
            return len(set(a) & set(b))
        
        n = len(properties)
        graph = defaultdict(list)
        
        # Build the graph based on the intersect condition
        for i in range(n):
            for j in range(i + 1, n):
                if intersect(properties[i], properties[j]) >= k:
                    graph[i].append(j)
                    graph[j].append(i)
        
        visited = [False] * n
        
        def dfs(node):
            stack = [node]
            while stack:
                curr = stack.pop()
                for neighbor in graph[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
        
        components = 0
        
        # Count connected components using DFS
        for i in range(n):
            if not visited[i]:
                visited[i] = True
                dfs(i)
                components += 1
        
        return components