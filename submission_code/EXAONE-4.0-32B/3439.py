from collections import deque
from typing import List

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        n = len(edges1) + 1
        m = len(edges2) + 1
        
        def get_diameter(edges, num_nodes):
            if num_nodes == 1:
                return 0
            graph = [[] for _ in range(num_nodes)]
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)
            
            dist = [-1] * num_nodes
            q = deque([0])
            dist[0] = 0
            while q:
                cur = q.popleft()
                for neighbor in graph[cur]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[cur] + 1
                        q.append(neighbor)
            
            u = 0
            for i in range(num_nodes):
                if dist[i] > dist[u]:
                    u = i
            
            dist = [-1] * num_nodes
            q.append(u)
            dist[u] = 0
            while q:
                cur = q.popleft()
                for neighbor in graph[cur]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[cur] + 1
                        q.append(neighbor)
            
            return max(dist)
        
        d1 = get_diameter(edges1, n)
        d2 = get_diameter(edges2, m)
        
        r1 = (d1 + 1) // 2
        r2 = (d2 + 1) // 2
        
        return max(d1, d2, r1 + r2 + 1)