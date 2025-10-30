from collections import deque
from typing import List

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        if not edges and n > 1:
            return -1
        
        max_weight = max(edge[2] for edge in edges)
        low, high = 1, max_weight
        answer = -1
        
        def is_possible(W):
            reversed_adj = [[] for _ in range(n)]
            for A, B, w in edges:
                if w <= W:
                    reversed_adj[B].append(A)
            in_degree = [0] * n
            q = deque()
            q.append(0)
            while q:
                u = q.popleft()
                for v in reversed_adj[u]:
                    if in_degree[v] < threshold:
                        in_degree[v] += 1
                        if in_degree[v] == 1:
                            q.append(v)
            for i in range(1, n):
                if in_degree[i] == 0:
                    return False
            return True
        
        while low <= high:
            mid = (low + high) // 2
            if is_possible(mid):
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return answer