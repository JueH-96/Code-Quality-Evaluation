from collections import deque
from typing import List

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        sorted_edges = sorted(edges, key=lambda x: x[2])
        low = 0
        high = 10**6
        ans = -1
        
        while low <= high:
            mid = (low + high) // 2
            graph = [[] for _ in range(n)]
            for u, v, w in sorted_edges:
                if w > mid:
                    break
                graph[v].append(u)
            
            visited = [False] * n
            queue = deque([0])
            visited[0] = True
            count = 1
            while queue:
                node = queue.popleft()
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        count += 1
                        queue.append(neighbor)
            
            if count == n:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return ans