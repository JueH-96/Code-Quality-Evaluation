import sys
from typing import List

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        sys.setrecursionlimit(10**7)
        n = len(edges) + 1
        # build adjacency list
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        
        dp0 = [0] * n  # dp0[u]: max sum in subtree u if we DO NOT keep edge to parent
        dp1 = [0] * n  # dp1[u]: max sum in subtree u if we DO keep edge to parent
        
        def dfs(u: int, parent: int):
            # We'll collect base = sum of dp0 over children
            # and gains for choosing to keep child-edges
            base = 0
            gains = []
            for v, w in adj[u]:
                if v == parent:
                    continue
                dfs(v, u)
                # if we don't pick u-v, we get dp0[v]
                base += dp0[v]
                # if we do pick u-v, we get dp1[v] + w instead of dp0[v]
                gains.append(dp1[v] + w - dp0[v])
            
            # sort gains descending
            gains.sort(reverse=True)
            
            # dp0: we may pick up to k child-edges
            take0 = 0
            for i in range(min(k, len(gains))):
                if gains[i] > 0:
                    take0 += gains[i]
                else:
                    break
            dp0[u] = base + take0
            
            # dp1: we keep the parent-edge, so can pick at most k-1 child-edges
            take1 = 0
            limit = k - 1
            if limit > 0:
                for i in range(min(limit, len(gains))):
                    if gains[i] > 0:
                        take1 += gains[i]
                    else:
                        break
            dp1[u] = base + take1
        
        # root at node 0, it has no parent so we use dp0[0]
        dfs(0, -1)
        return dp0[0]