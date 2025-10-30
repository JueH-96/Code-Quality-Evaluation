from typing import List
import sys
sys.setrecursionlimit(10**6)

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        # Build the tree as an adjacency list.
        n = len(edges) + 1
        tree = [[] for _ in range(n)]
        for u, v, w in edges:
            tree[u].append((v, w))
            tree[v].append((u, w))
        
        # We will use DFS to compute two values for every node u:
        #   g(u): maximum sum obtainable in the subtree rooted at u,
        #         when u is NOT connected to its parent
        #         (so u can use up to k edges to its children)
        #   h(u): maximum sum obtainable in the subtree rooted at u,
        #         when u IS connected to its parent
        #         (so u already uses one incident edge from its parent,
        #          and can only choose up to k-1 edges to its children)
        #
        # The recurrence works as follows: For each child v of u (in the DFS tree)
        # with the connecting edge weight w, if we do NOT choose the edge (u,v)
        # then the best from subtree v is assumed to be g(v) (v is not connected
        # from u, so it can use full capacity k for connecting its own children).
        # If we DO choose (u,v) then we add the weight w and v becomes "attached"
        # so its available capacity is k-1, contributing h(v).
        # Hence the extra gain (or difference) of using edge (u,v) over not using it
        # is: diff = [w + h(v)] - g(v).
        # However, at node u we are allowed to choose at most X edges among its children,
        # where X = k if u is not attached, or X = k-1 if u is attached.
        # Since the decisions for different children do not interact except for this cap on count,
        # we can compute:
        #    g(u) = sum_{child v} g(v) + sum of top min(k, count) positive differences among children.
        #    h(u) = sum_{child v} g(v) + sum of top min(k-1, count) positive differences among children.
        #
        # Base case: a leaf returns 0 for both.
        
        def dfs(u: int, parent: int) -> (int, int):
            sums = 0
            diffs = []
            # Process all neighbors (children)
            for v, w in tree[u]:
                if v == parent:
                    continue
                # Recurse on child
                g_v, h_v = dfs(v, u)
                sums += g_v
                diff = (w + h_v) - g_v
                diffs.append(diff)
            # Sort diffs in descending order
            diffs.sort(reverse=True)
            # For node u when not attached: capacity = k edges can be chosen among children.
            g_u = sums
            # choose up to k positive diffs
            for i in range(min(k, len(diffs))):
                if diffs[i] > 0:
                    g_u += diffs[i]
                else:
                    break
            # For node u when attached: capacity = k - 1.
            h_u = sums
            for i in range(min(k - 1, len(diffs))):
                if diffs[i] > 0:
                    h_u += diffs[i]
                else:
                    break
            return g_u, h_u
        
        # We can choose any node as root. For the root, it is not attached to a parent.
        ans, _ = dfs(0, -1)
        return ans