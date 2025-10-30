from collections import defaultdict
from typing import List

class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)
        
        def intersect(a, b):
            return len(set(a) & set(b))
        
        graph = defaultdict(list)
        for i in range(n):
            for j in range(i+1, n):
                if intersect(properties[i], properties[j]) >= k:
                    graph[i].append(j)
                    graph[j].append(i)
        
        def dfs(node, visited):
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, visited)
        
        visited = set()
        components = 0
        for i in range(n):
            if i not in visited:
                components += 1
                dfs(i, visited)
        
        return components