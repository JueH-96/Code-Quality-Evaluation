from typing import List
from collections import deque

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        reversed_edges = []
        max_w = 0
        for a, b, w in edges:
            reversed_edges.append((b, a, w))
            max_w = max(max_w, w)
        
        left, right = 1, max_w
        ans = -1
        
        while left <= right:
            mid = (left + right) // 2
            # Build adjacency list for reversed edges with weight <= mid
            adj = [[] for _ in range(n)]
            for u, v, w in reversed_edges:
                if w <= mid:
                    adj[u].append(v)
            
            # BFS to check reachability from node 0
            visited = [False] * n
            queue = deque([0])
            visited[0] = True
            count = 1
            
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        count += 1
                        queue.append(v)
            
            if count == n:
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return ans