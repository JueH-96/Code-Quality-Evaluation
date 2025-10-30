from typing import List
import sys
sys.setrecursionlimit(10**7)

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # weight to enter node i
        w = [1 if (i & 1) else 2 for i in range(n)]
        parent = [-1] * n
        # get a DFS preorder to set up parent links
        order = []
        stack = [0]
        parent[0] = -1
        while stack:
            u = stack.pop()
            order.append(u)
            for v in adj[u]:
                if v == parent[u]:
                    continue
                parent[v] = u
                stack.append(v)
        # dp_down[u] = max distance from u down into its subtree
        dp_down = [0] * n
        for u in reversed(order):
            best = 0
            for v in adj[u]:
                if v == parent[u]:
                    continue
                # going u -> v costs w[v], then dp_down[v]
                cand = w[v] + dp_down[v]
                if cand > best:
                    best = cand
            dp_down[u] = best
        # dp_up[u] = max distance from u going up/outside its subtree
        dp_up = [0] * n
        # propagate dp_up in preorder
        for u in order:
            # find top two child contributions at u
            max1 = 0
            max2 = 0
            for v in adj[u]:
                if v == parent[u]:
                    continue
                c = w[v] + dp_down[v]
                if c >= max1:
                    max2 = max1
                    max1 = c
                elif c > max2:
                    max2 = c
            # now set dp_up for each child
            for v in adj[u]:
                if v == parent[u]:
                    continue
                c = w[v] + dp_down[v]
                # best contribution excluding this child
                best_excl = max2 if c == max1 else max1
                # two possibilities: go up from u, or go from u down another child
                cand1 = w[u] + dp_up[u]
                cand2 = w[u] + best_excl
                dp_up[v] = cand1 if cand1 >= cand2 else cand2
        # answer is max of down and up for each node
        res = [max(dp_down[i], dp_up[i]) for i in range(n)]
        return res