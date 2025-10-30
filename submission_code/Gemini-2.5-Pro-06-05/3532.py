import sys
from typing import List

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        
        # Ensure recursion limit is sufficient for deep trees.
        # This is good practice, although LeetCode's environment might have a high limit.
        if n > sys.getrecursionlimit():
            sys.setrecursionlimit(n + 50)
            
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # d1[u]: longest path time starting at u, going "down" into its subtree.
        # d2[u]: second longest path time starting at u, going "down".
        d1 = [0] * n
        d2 = [0] * n
        
        # The time cost to mark a node v depends on its parity.
        def cost(node_v: int) -> int:
            return 2 if node_v % 2 == 0 else 1

        # DFS 1 (post-order): Computes longest and second-longest downward paths from each node.
        def dfs1(u: int, p: int):
            for v in adj[u]:
                if v == p:
                    continue
                
                dfs1(v, u)
                
                # Path from u to a leaf in v's subtree has time: cost(v) + d1[v]
                path_len = cost(v) + d1[v]
                
                if path_len > d1[u]:
                    d2[u] = d1[u]
                    d1[u] = path_len
                elif path_len > d2[u]:
                    d2[u] = path_len
        
        # We arbitrarily root the tree at node 0.
        dfs1(0, -1)

        # ans[u]: final answer for node u.
        # up[u]: longest path time starting at u, going "upwards".
        ans = [0] * n
        up = [0] * n

        # DFS 2 (pre-order): Computes longest upward path and the final answer for each node.
        def dfs2(u: int, p: int):
            # The longest path from u is the max of the longest downward and upward paths.
            ans[u] = max(d1[u], up[u])
            
            for v in adj[u]:
                if v == p:
                    continue
                
                # This is the length of the downward path from u that goes through v.
                path_from_u_down_v_branch = cost(v) + d1[v]
                
                # We need the longest downward path from u that does NOT go through v.
                if path_from_u_down_v_branch == d1[u]:
                    sibling_path_len = d2[u]
                else:
                    sibling_path_len = d1[u]

                # up[v] = time(v->u) + max(time_from_u_up, time_from_u_down_other_branch)
                # time(v->u) is cost(u).
                up[v] = cost(u) + max(up[u], sibling_path_len)
                
                dfs2(v, u)

        # Start the second DFS from the root. For the root, the upward path length is 0.
        dfs2(0, -1)
        
        return ans