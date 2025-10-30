from collections import deque
from typing import List

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        # Extract and sort unique weights
        weights = sorted({w for _, _, w in edges})
        if not weights:
            return -1
        
        # Build list of edges once
        # edges are (u, v, w)
        
        def can_reach_all(max_w: int) -> bool:
            # Build reverse adjacency for edges with weight <= max_w
            rev_adj = [[] for _ in range(n)]
            for u, v, w in edges:
                if w <= max_w:
                    # edge u->v in original, so v->u in reverse
                    rev_adj[v].append(u)
            # BFS from node 0 on reverse graph
            seen = [False] * n
            dq = deque([0])
            seen[0] = True
            cnt = 1
            while dq:
                x = dq.popleft()
                for y in rev_adj[x]:
                    if not seen[y]:
                        seen[y] = True
                        cnt += 1
                        dq.append(y)
            return cnt == n
        
        # Binary search on weights
        lo, hi = 0, len(weights) - 1
        ans = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_reach_all(weights[mid]):
                ans = weights[mid]
                hi = mid - 1
            else:
                lo = mid + 1
        
        return ans