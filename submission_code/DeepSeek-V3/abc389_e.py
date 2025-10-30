from typing import List
from collections import deque

class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)
        # Preprocess each property to a set of distinct elements
        sets = [set(arr) for arr in properties]
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                common = len(sets[i] & sets[j])
                if common >= k:
                    adj[i].append(j)
                    adj[j].append(i)
        
        visited = [False] * n
        components = 0
        
        for i in range(n):
            if not visited[i]:
                # BFS or DFS to mark all connected nodes
                queue = deque([i])
                visited[i] = True
                while queue:
                    node = queue.popleft()
                    for neighbor in adj[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                components += 1
        return components