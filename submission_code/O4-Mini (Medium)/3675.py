import sys
from typing import List

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        sys.setrecursionlimit(10**7)
        n = len(edges) + 1
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        
        # Build a parent array and a DFS order
        parent = [-1] * n
        parent[0] = 0  # mark root's parent as itself
        order = []
        stack = [0]
        while stack:
            u = stack.pop()
            order.append(u)
            for v, w in adj[u]:
                if parent[v] == -1:
                    parent[v] = u
                    stack.append(v)
        
        # dp0[u]: best sum in subtree u if we do NOT keep the edge to parent
        # dp1[u]: best sum in subtree u if we DO keep the edge to parent
        dp0 = [0] * n
        dp1 = [0] * n
        NEG_INF = -10**30
        
        # Process nodes in reverse DFS order (post-order)
        for u in reversed(order):
            base = 0
            diffs = []
            # Gather contributions from children
            for v, w in adj[u]:
                if parent[v] == u:
                    # v is a child of u
                    base += dp0[v]
                    # If we pick edge (u,v), gain w + dp1[v] instead of dp0[v]
                    diff = w + dp1[v] - dp0[v]
                    diffs.append(diff)
            
            # Sort diffs descending
            diffs.sort(reverse=True)
            
            # Compute dp0[u]: up to k child-edges
            dp0[u] = base
            taken = 0
            for d in diffs:
                if taken >= k or d <= 0:
                    break
                dp0[u] += d
                taken += 1
            
            # Compute dp1[u]: if we also keep the edge to parent, we can take up to k-1 child-edges
            if k > 0:
                dp1[u] = base
                taken = 0
                for d in diffs:
                    if taken >= k - 1 or d <= 0:
                        break
                    dp1[u] += d
                    taken += 1
            else:
                dp1[u] = NEG_INF
        
        # The root has no parent-edge, so answer is dp0[0]
        return dp0[0]