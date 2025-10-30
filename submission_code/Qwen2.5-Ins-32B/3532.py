from typing import List

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        from collections import defaultdict
        
        def dfs(node, parent, depth):
            if node % 2 == 0:
                even_max = 0
                odd_max = 0
                for neighbor in graph[node]:
                    if neighbor != parent:
                        even, odd = dfs(neighbor, node, depth + 1)
                        even_max = max(even_max, even)
                        odd_max = max(odd_max, odd)
                return (max(even_max, depth), max(odd_max, depth + 1))
            else:
                even_max = 0
                odd_max = 0
                for neighbor in graph[node]:
                    if neighbor != parent:
                        even, odd = dfs(neighbor, node, depth + 1)
                        even_max = max(even_max, even)
                        odd_max = max(odd_max, odd)
                return (max(even_max, depth + 1), max(odd_max, depth))
        
        n = len(edges) + 1
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        result = []
        for i in range(n):
            even, odd = dfs(i, -1, 0)
            result.append(max(even, odd))
        
        return result