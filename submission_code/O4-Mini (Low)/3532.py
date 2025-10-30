from typing import List
import sys
sys.setrecursionlimit(10**7)

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        # Build adjacency
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        # Weight of entering node: odd ->1, even->2
        w = [1 if (i & 1) else 2 for i in range(n)]
        # We'll root the tree at 0, compute dp_down and dp_up
        parent = [-1]*n
        order = []
        # iterative DFS to get parent and post-order
        stack = [0]
        parent[0] = 0
        while stack:
            u = stack.pop()
            order.append(u)
            for v in g[u]:
                if parent[v] == -1:
                    parent[v] = u
                    stack.append(v)
        # now order is a preorder; we'll process in reverse for post-order
        dp_down = [0]*n
        # best child contributions: best1, best2 for each node
        best1 = [0]*n
        best2 = [0]*n
        for u in reversed(order):
            # look at children v of u
            for v in g[u]:
                if v == parent[u]:
                    continue
                # contribution from v
                c = w[v] + dp_down[v]
                # update u's two bests
                if c >= best1[u]:
                    best2[u] = best1[u]
                    best1[u] = c
                elif c > best2[u]:
                    best2[u] = c
            # dp_down[u] is simply best1[u]
            dp_down[u] = best1[u]
        # dp_up
        dp_up = [0]*n
        # now do a BFS/DFS from root to compute dp_up for children
        stack = [0]
        while stack:
            u = stack.pop()
            for v in g[u]:
                if v == parent[u]:
                    continue
                # pick the best among siblings: if best1[u] came from v then use best2[u]
                use_sibling = best1[u]
                if w[v] + dp_down[v] == best1[u]:
                    use_sibling = best2[u]
                # dp_up[v] = w[u] + max(dp_up[u], use_sibling)
                dp_up[v] = w[u] + max(dp_up[u], use_sibling)
                stack.append(v)
        # answer for each i is max(dp_down[i], dp_up[i])
        ans = [max(dp_down[i], dp_up[i]) for i in range(n)]
        return ans

# For testing purposes
if __name__ == "__main__":
    sol = Solution()
    print(sol.timeTaken([[0,1],[0,2]]))        # [2,4,3]
    print(sol.timeTaken([[0,1]]))              # [1,2]
    print(sol.timeTaken([[2,4],[0,1],[2,3],[0,2]]))  # [4,6,3,5,5]