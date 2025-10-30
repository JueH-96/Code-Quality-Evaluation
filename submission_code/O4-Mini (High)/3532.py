from typing import List

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        # build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # weight of a node: 1 if odd, 2 if even
        w = [2] * n
        for i in range(1, n, 2):
            w[i] = 1

        parent = [-1] * n
        down = [0] * n
        # for each node u, best1_val[u] = largest child‐contribution
        # best2_val[u] = second largest, best1_child[u] = which child gave best1_val
        best1_val = [0] * n
        best2_val = [0] * n
        best1_child = [-1] * n

        # Post-order DFS (iterative) to compute down[u]
        stack = [(0, -1, False)]
        while stack:
            u, p, visited = stack.pop()
            if not visited:
                parent[u] = p
                stack.append((u, p, True))
                for v in adj[u]:
                    if v == p:
                        continue
                    stack.append((v, u, False))
            else:
                max1 = 0
                max2 = 0
                bch = -1
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    val = w[v] + down[v]
                    if val > max1:
                        max2 = max1
                        max1 = val
                        bch = v
                    elif val > max2:
                        max2 = val
                down[u] = max1
                best1_val[u] = max1
                best2_val[u] = max2
                best1_child[u] = bch

        # Pre-order DFS to compute up[u]
        up = [0] * n
        stack = [0]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v == parent[u]:
                    continue
                # pick best among (1) up[u] and (2) best sibling contribution
                if best1_child[u] != v:
                    sib = best1_val[u]
                else:
                    sib = best2_val[u]
                best_excl = up[u] if up[u] > sib else sib
                up[v] = w[u] + best_excl
                stack.append(v)

        # The time for each i is the max of down[i] and up[i]
        res = [0] * n
        for i in range(n):
            res[i] = down[i] if down[i] > up[i] else up[i]
        return res