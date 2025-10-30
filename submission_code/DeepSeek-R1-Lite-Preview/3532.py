from typing import List

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        from collections import defaultdict

        n = len(edges) + 1
        tree = defaultdict(list)
        for u, v in edges:
            tree[u].append(v)
            tree[v].append(u)

        def dfs(root, parent, sum_inc, depth, is_root):
            for child in tree[root]:
                if child != parent:
                    increment = 1 if child % 2 == 1 else 2
                    sum_inc[child] = sum_inc[root] + increment
                    dfs(child, root, sum_inc, depth + 1, False)

        # First DFS from node 0
        sum_inc0 = [0] * n
        dfs(0, -1, sum_inc0, 0, True)
        # Find node A with maximum sum_inc from node 0
        A = sum_inc0.index(max(sum_inc0))
        # Second DFS from node A
        sum_incA = [0] * n
        dfs(A, -1, sum_incA, 0, True)
        # Find node B with maximum sum_inc from node A
        B = sum_incA.index(max(sum_incA))
        
        # Now, for each root, the maximum marking time is the max of sum_inc from root to A and root to B
        times = [0] * n
        for u in range(n):
            times[u] = max(sum_inc0[u] + sum_incA[B] - sum_incA[u], sum_incA[u] + sum_inc0[A] - sum_inc0[u])
        
        return times