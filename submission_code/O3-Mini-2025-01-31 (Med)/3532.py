from typing import List
import sys
sys.setrecursionlimit(10**6)

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        # Build the undirected tree as an adjacency list.
        n = len(edges) + 1
        graph = [[] for _ in range(n)]
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
            
        # Define the cost function:
        # When a node v becomes marked by a neighbor,
        # the time difference is:
        #  1 if v is odd, 2 if v is even.
        def cost(v: int) -> int:
            return 1 if (v & 1) == 1 else 2
        
        # dp[u] will store the maximum distance from u to any node in its subtree (in a DFS-tree sense)
        dp = [0] * n
        
        # We'll build a tree structure with a chosen root (say 0)
        parent = [-1] * n
        
        # First DFS: compute dp[u] for every node (only considering children in our DFS tree)
        def dfs(u: int, par: int):
            parent[u] = par
            best = 0
            for w in graph[u]:
                if w == par:
                    continue
                dfs(w, u)
                # When going from u to child w, the increment is determined by the child's cost.
                candidate = cost(w) + dp[w]
                if candidate > best:
                    best = candidate
            dp[u] = best
        
        dfs(0, -1)
        
        # up[u] will store the maximum distance from u going outside its subtree (toward parent's side)
        up = [0] * n
        ans = [0] * n
        
        # Second DFS: re-root technique.
        def dfs2(u: int, par: int):
            # The best distance from u overall is either from inside its subtree (dp[u]) or coming from above (up[u])
            ans[u] = max(dp[u], up[u])
            
            # Precompute the candidate values from children of u.
            # For each child v, candidate = cost(v) + dp[v]
            children = []
            for w in graph[u]:
                if w == par:
                    continue
                children.append(w)
            m = len(children)
            cand = [0] * m
            for i, v in enumerate(children):
                cand[i] = cost(v) + dp[v]
            
            # Build prefix and suffix maximum arrays.
            prefix = [0] * m
            suffix = [0] * m
            for i in range(m):
                if i == 0:
                    prefix[i] = cand[i]
                else:
                    prefix[i] = max(prefix[i-1], cand[i])
            for i in range(m-1, -1, -1):
                if i == m-1:
                    suffix[i] = cand[i]
                else:
                    suffix[i] = max(suffix[i+1], cand[i])
            
            for i, v in enumerate(children):
                # best candidate from parent's children excluding v
                best_exclude = 0
                if i > 0:
                    best_exclude = prefix[i-1]
                if i < m - 1:
                    best_exclude = max(best_exclude, suffix[i+1])
                # When going from v to u, the cost is determined by u's cost.
                # So for v, the candidate from parent's side:
                #   up_candidate = cost(u) + max( up[u], best_exclude )
                candidate = cost(u) + max(up[u], best_exclude)
                up[v] = candidate
                dfs2(v, u)
        
        dfs2(0, -1)
        return ans