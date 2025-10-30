from typing import List
import sys

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        # Increase recursion limit because n can be as high as 10^5
        sys.setrecursionlimit(10**6)
        
        n = len(edges) + 1
        # Build an undirected tree (adjacency list)
        graph = [[] for _ in range(n)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        # For each node, define its “cost” (marking time delay factor)
        # If node is odd: cost = 1, if even: cost = 2.
        c = [1 if (i % 2) == 1 else 2 for i in range(n)]
        
        # We want to compute, for each starting node s,
        # the time when every node is marked.
        # Our observation:
        #   When s is marked at time 0, each other node v along the unique s->v path 
        #   gets marked at time = sum_{w in path from s (excluded) to v} c(w)
        # So the "spread‐time" from s is:
        #   T(s) = max_{v in V} (sum_{w in path from s to v} c(w)).
        #
        # We can “simulate” this process as a shortest–path (or, here, longest–path)
        # problem in a tree with directional edge cost:
        #   When going from u to v, cost = c(v)
        # That is, for a given path s -> v1 -> v2 -> ... -> vk = v,
        # the total time = c(v1) + c(v2) + ... + c(v).
        #
        # Although the weight on an edge depends on the target,
        # note that along the unique path this sum is fixed.
        #
        # To facilitate a re‐rooting DP we “shift” the cost,
        # rewriting the accumulation in a symmetric way.
        #
        # Multiply everything by 2 to eliminate fractions.
        # Define a new function:
        #    f2(s) = max_{v in V} [d_w(s, v) + c(v)]
        # where d_w(s,v) is the sum over edges along the s->v path with
        #   for edge (u,v): W(u,v) = c(u) + c(v)
        # Notice that if v == s the sum is zero and f2(s) >= c(s).
        # Then the answer we want (the maximum time when all nodes are marked)
        # is:
        #    ans[s] = (f2(s) - c(s)) // 2.
        #
        # We now compute for every vertex s the value:
        #    f2(s) = max_{v in V} [ d_w(s, v) + c(v) ]
        #
        # Because the tree is undirected (and now the edge weights W(u,v) = c(u)+c(v) are symmetric)
        # we can compute f2(s) by doing a “re‐rooting” DP.
        #
        # We define dp[u] = maximum value of (d_w(u, x) + c(x)) restricted to 
        # x in the subtree of u (when the tree is rooted arbitrarily).
        # Notice that x = u is allowed (giving dp[u] >= c(u)).
        # Then for a child v of u the candidate value is:
        #    candidate = (c(u) + c(v)) + dp[v]
        # and we take:
        #    dp[u] = max( c(u),  max_{child v: u->v}( (c(u) + c(v)) + dp[v] ) ).
        #
        # Then we do an “upward” (re‐rooting) DP to let every node also consider paths
        # that go up through the parent. For each vertex u we maintain up[u] equal to the best 
        # candidate coming from ancestors (i.e. outside of u’s subtree).
        #
        # Finally, for each vertex u, define
        #   f2(u) = max(dp[u], up[u])
        # and then our answer is
        #   ans[u] = (f2(u) - c(u)) // 2.
        
        # We'll use -INF as a very small number.
        INF = -10**9
        
        dp = [0] * n         # dp[u] = best value from u's subtree (in our f2 metric, multiplied by 1)
        up = [INF] * n       # up[u] = best value coming from above (parent side)
        # For re-rooting we also record for each node u:
        # best1[u] = highest candidate among children: candidate = (c(u)+ c(v)) + dp[v]
        # best2[u] = second highest candidate among children
        # bestChild[u] = child v that gives best1[u]
        best1 = [INF] * n
        best2 = [INF] * n
        bestChild = [-1] * n
        
        # First DFS: compute dp[] and record best candidates from children.
        def dfs1(u: int, parent: int) -> None:
            dp[u] = c[u]  # Base: path of length 0: from u to itself gives value c(u).
            for v in graph[u]:
                if v == parent:
                    continue
                dfs1(v, u)
                # Candidate if we use branch from u to v: 
                # d_w(u,v) is c(u)+ c(v) and then add best down from v.
                candidate = c[u] + c[v] + dp[v]
                if candidate > dp[u]:
                    dp[u] = candidate
                # Update best1 and best2 for children branch from u
                if candidate > best1[u]:
                    best2[u] = best1[u]
                   	best1[u] = candidate
                    bestChild[u] = v
                elif candidate > best2[u]:
                    best2[u] = candidate
        
        dfs1(0, -1)
        
        # Second DFS: propagate parent's contribution (up[]) to children.
        def dfs2(u: int, parent: int) -> None:
            for v in graph[u]:
                if v == parent:
                    continue
                # When moving from u to v, we must choose the best candidate from u that does
                # not come from v.
                if bestChild[u] == v:
                    # exclude the branch from v: candidate = max(up[u], c[u], best2[u])
                    cand = max(up[u], c[u], best2[u])
                else:
                    # When v is not the branch that gave dp[u] we may use dp[u].
                    cand = max(up[u], dp[u])
                # When we go from u to child v, we add the edge cost (c(u)+ c(v)).
                up[v] = cand + c[u] + c[v]
                dfs2(v, u)
                
        # For root (node 0) there is no parent's contribution; set up[0] to a very small value.
        dfs2(0, -1)
        
        # Now for each vertex, the best overall candidate computed is f2[u] = max(dp[u], up[u]).
        # Then the answer for vertex u is: (f2[u] - c[u]) // 2.
        ans = [0] * n
        for i in range(n):
            f2_val = dp[i] if dp[i] >= up[i] else up[i]
            ans2 = f2_val - c[i]
            # Because we multiplied all distances by 1 (effectively having no fractions now)
            # and originally the marking time from s would be (f2(s)- c(s)) / 2,
            # we can perform integer division (it will be an integer because
            # the original sum c(w) is integer).
            ans[i] = ans2 // 2
        return ans