from typing import List
import sys
sys.setrecursionlimit(300000)


class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        """
        For every node i (taken as the only initially marked node, t = 0) we have to
        know the moment when the very last node of the tree becomes marked.

        Observation
        -----------
        Let w(v) be
              1  if v is odd
              2  if v is even

        If we are at node u at time T, an adjacent node v will be marked at

              T + w(v)

        (because w(v) is exactly the waiting time required for v).
        Hence, starting from a source s, the instant a node x is marked equals

              sum of w(.) along the unique path (s … x)                  (*)

        where the weight of the first vertex (the source s itself) is **not**
        included.  
        Therefore, for a fixed source s, the time when every node is marked is

              max_{x in V}   dist_s(x)

        where dist_s(x) is given by (*).

        The task is thus to compute, for every vertex s, the maximum weighted
        distance that starts at s – the eccentricity w.r.t. the directed
        distances defined by w.

        Tree DP
        -------
        Treat the tree as rooted (root = 0).  
        For every vertex u keep

        down[u]  – the largest weighted distance that starts at u and stays
                   completely inside the subtree of u.

        best1[u], best2[u] – the largest and the second largest values of
                   w(child) + down[child] among the children of u
                   (needed for re-rooting).

        First DFS (post-order) sets the above values.

        In the second DFS we propagate results to the children by keeping

        up[u]   – the largest weighted distance that starts at u and reaches
                  vertices **outside** u's own subtree.

        For a child v of u

            up[v] = w(u) + max( up[u],
                                best value coming from another child of u )

        Finally

            answer[u] = max( down[u], up[u] )

        Complexity
        ----------
        Two linear passes ⇒  O(n) time and O(n) memory.
        """
        n = len(edges) + 1
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # vertex weights
        w = [1 if i & 1 else 2 for i in range(n)]

        down = [0] * n
        best1 = [0] * n        # largest child contribution
        best2 = [0] * n        # second largest child contribution

        # ---------- first DFS : compute down / best1 / best2 ----------
        def dfs1(u: int, p: int) -> None:
            for v in adj[u]:
                if v == p:
                    continue
                dfs1(v, u)
                cand = w[v] + down[v]
                if cand > best1[u]:
                    best2[u] = best1[u]
                    best1[u] = cand
                elif cand > best2[u]:
                    best2[u] = cand
            down[u] = best1[u]

        dfs1(0, -1)

        # ---------- second DFS : compute up ----------
        up = [0] * n           # root has no ancestor => 0

        def dfs2(u: int, p: int) -> None:
            for v in adj[u]:
                if v == p:
                    continue
                cand = w[v] + down[v]
                # choose among children other than v
                use = best1[u] if cand != best1[u] else best2[u]
                up[v] = w[u] + max(up[u], use)
                dfs2(v, u)

        dfs2(0, -1)

        # ---------- final answer ----------
        return [max(down[i], up[i]) for i in range(n)]