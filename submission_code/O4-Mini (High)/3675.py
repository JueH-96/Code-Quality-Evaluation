from typing import List

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        # Number of nodes
        n = len(edges) + 1
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        
        # Build a rooted tree at node 0 using iterative DFS
        parent = [-1] * n
        parent[0] = 0       # mark root as visited
        pw = [0] * n        # weight of edge from node to its parent
        children = [[] for _ in range(n)]
        
        stack = [0]
        order = []  # will hold nodes in preorder
        
        while stack:
            u = stack.pop()
            order.append(u)
            for v, w in adj[u]:
                if parent[v] == -1:
                    parent[v] = u
                    pw[v] = w
                    children[u].append(v)
                    stack.append(v)
        
        # dp0[u] = max sum in subtree u if the edge to parent is NOT taken
        # dp1[u] = max sum in subtree u if the edge to parent IS taken
        dp0 = [0] * n
        dp1 = [0] * n
        
        # Process nodes in post-order (children before parent)
        for u in reversed(order):
            base = 0
            gains = []
            # First, assume we do NOT take any child-edge: sum up dp0 of children
            for v in children[u]:
                base += dp0[v]
                # If we do take the edge (u,v), gain is dp1[v] + w(u,v) - dp0[v]
                gains.append(dp1[v] + pw[v] - dp0[v])
            
            # Start with no extra child-edges
            dp0[u] = base
            dp1[u] = base
            
            if gains:
                # Sort gains descending
                gains.sort(reverse=True)
                # For dp0[u], we can pick up to k positive gains
                cnt0 = k
                extra0 = 0
                for i in range(min(cnt0, len(gains))):
                    if gains[i] > 0:
                        extra0 += gains[i]
                    else:
                        break
                dp0[u] = base + extra0
                
                # For dp1[u], since parent-edge is taken, we can pick up to k-1 child-edges
                cnt1 = k - 1
                extra1 = 0
                if cnt1 > 0:
                    for i in range(min(cnt1, len(gains))):
                        if gains[i] > 0:
                            extra1 += gains[i]
                        else:
                            break
                dp1[u] = base + extra1
        
        # The root has no parent-edge, so the answer is dp0[0]
        return dp0[0]