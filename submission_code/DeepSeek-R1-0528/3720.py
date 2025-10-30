from collections import deque
from typing import List

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        max_weight = 0
        for edge in edges:
            if edge[2] > max_weight:
                max_weight = edge[2]
        
        low, high = 1, max_weight
        ans = -1
        
        def check(X):
            rev_out = [[] for _ in range(n)]
            for a, b, w in edges:
                if w <= X:
                    rev_out[b].append(a)
            
            active = set([0])
            active_parents = [set() for _ in range(n)]
            for a in rev_out[0]:
                if a != 0:
                    active_parents[a].add(0)
            
            connected = [False] * n
            connected[0] = True
            deg_out = [0] * n
            queue = deque()
            
            for v in range(1, n):
                if active_parents[v]:
                    queue.append(v)
            
            while queue:
                v = queue.popleft()
                if connected[v] or not active_parents[v]:
                    continue
                u = next(iter(active_parents[v]))
                connected[v] = True
                deg_out[u] += 1
                if deg_out[u] == threshold:
                    active.remove(u)
                    for a in rev_out[u]:
                        if u in active_parents[a]:
                            active_parents[a].discard(u)
                active.add(v)
                for a in rev_out[v]:
                    if not connected[a]:
                        if v not in active_parents[a]:
                            active_parents[a].add(v)
                            queue.append(a)
            return all(connected)
        
        while low <= high:
            mid = (low + high) // 2
            if check(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return ans