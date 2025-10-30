from typing import List

class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)
        # Create sets for each property to handle distinct elements
        sets_prop = [set(prop) for prop in properties]
        
        # Build the adjacency list
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if len(sets_prop[i] & sets_prop[j]) >= k:
                    adj[i].append(j)
                    adj[j].append(i)
        
        # DFS to find connected components
        visited = [False] * n
        components = 0
        
        def dfs(node):
            visited[node] = True
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    dfs(neighbor)
        
        for i in range(n):
            if not visited[i]:
                components += 1
                dfs(i)
        
        return components