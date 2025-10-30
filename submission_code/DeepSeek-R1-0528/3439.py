import collections
from typing import List

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def compute(edges):
            n = len(edges) + 1
            graph = [[] for _ in range(n)]
            for a, b in edges:
                graph[a].append(b)
                graph[b].append(a)
            
            if n == 1:
                return (0, 0)
            
            dist0 = [-1] * n
            q = collections.deque([0])
            dist0[0] = 0
            u = 0
            while q:
                cur = q.popleft()
                if dist0[cur] > dist0[u]:
                    u = cur
                for neighbor in graph[cur]:
                    if dist0[neighbor] == -1:
                        dist0[neighbor] = dist0[cur] + 1
                        q.append(neighbor)
            
            dist_u = [-1] * n
            q.append(u)
            dist_u[u] = 0
            v = u
            while q:
                cur = q.popleft()
                if dist_u[cur] > dist_u[v]:
                    v = cur
                for neighbor in graph[cur]:
                    if dist_u[neighbor] == -1:
                        dist_u[neighbor] = dist_u[cur] + 1
                        q.append(neighbor)
            diameter = dist_u[v]
            
            dist_v = [-1] * n
            q.append(v)
            dist_v[v] = 0
            while q:
                cur = q.popleft()
                for neighbor in graph[cur]:
                    if dist_v[neighbor] == -1:
                        dist_v[neighbor] = dist_v[cur] + 1
                        q.append(neighbor)
            
            ecc = [max(dist_u[i], dist_v[i]) for i in range(n)]
            min_ecc = min(ecc)
            return (diameter, min_ecc)
        
        d1, min_ecc1 = compute(edges1)
        d2, min_ecc2 = compute(edges2)
        return max(d1, d2, min_ecc1 + min_ecc2 + 1)