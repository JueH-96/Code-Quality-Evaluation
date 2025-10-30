import collections
from typing import List

class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)
        
        # Helper function to calculate intersection size of distinct elements
        def intersect(a: List[int], b: List[int]) -> int:
            set_a = set(a)
            set_b = set(b)
            return len(set_a.intersection(set_b))
            # Or more concisely: return len(set(a) & set(b))

        # Build the graph using an adjacency list
        adj = collections.defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n): # Avoid self-loops and duplicate edges (i,j) vs (j,i)
                if intersect(properties[i], properties[j]) >= k:
                    adj[i].append(j)
                    adj[j].append(i)

        # Find connected components using DFS
        visited = [False] * n
        component_count = 0

        def dfs(node):
            visited[node] = True
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    dfs(neighbor)
                    
        # Alternative using BFS
        # def bfs(start_node):
        #     queue = collections.deque([start_node])
        #     visited[start_node] = True
        #     while queue:
        #         u = queue.popleft()
        #         for v in adj[u]:
        #             if not visited[v]:
        #                 visited[v] = True
        #                 queue.append(v)

        # Iterate through all nodes to find unvisited ones and start traversal
        for i in range(n):
            if not visited[i]:
                component_count += 1
                dfs(i) # Or bfs(i)

        return component_count