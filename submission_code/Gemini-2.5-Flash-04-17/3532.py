import sys
sys.setrecursionlimit(200000) # Increase recursion depth

from typing import List

class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1

        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Calculate cost for each node
        # Odd nodes cost 1, Even nodes cost 2
        cost_val = [1 if i % 2 != 0 else 2 for i in range(n)]

        # DP state: max path sum downwards from u (excluding u)
        # dp_down[u] = max_{v in subtree(u)} path_dist(u, v)
        # where path_dist(u, v) = sum(cost_val[w] for w on path u..v, w != u)
        dp_down = [0] * n

        # DP state: max path sum upwards from u (excluding u)
        # dp_up[u] = max_{v not in subtree(u)} path_dist(u, v)
        dp_up = [0] * n

        # Helper arrays for efficient max/second max children values in DFS1
        # Stores max and second max values of (cost_val[c] + dp_down[c]) for child c of u
        # (cost_val[c] + dp_down[c]) represents max path_dist(u, v) for v in subtree(c) via u->c path
        child_down_max1 = [0] * n
        child_down_max2 = [0] * n
        # Stores the child node achieving the max value (needed to find sibling max in DFS2)
        child_down_max1_node = [-1] * n

        # DFS1: Compute dp_down and child_down_max1/max2/node
        # This is a post-order traversal (children results needed for parent)
        def dfs1(u: int, p: int):
            max_from_child_subtree = 0
            child_down_list = [] # Store (value, child_node) for children's downward paths

            for v in adj[u]:
                if v != p:
                    dfs1(v, u)
                    # Value for child v: max path_dist(u, x) for x in subtree(v) via u->v path
                    # path_dist(u, x) = cost_val[v] + path_dist(v, x)
                    # Max over x in subtree(v) is cost_val[v] + max_{x in subtree(v)} path_dist(v, x)
                    # max_{x in subtree(v)} path_dist(v, x) is dp_down[v] by definition
                    child_val = cost_val[v] + dp_down[v]
                    child_down_list.append((child_val, v))
                    max_from_child_subtree = max(max_from_child_subtree, child_val)

            # dp_down[u] = max path_dist(u, v) for v in subtree(u)
            # v can be u (path_dist(u,u) = 0) or v in a child subtree.
            dp_down[u] = max(0, max_from_child_subtree) # max(path_dist(u,u), max over child subtrees)

            # Store max and second max child values for current node u
            # These values are the max path_dist(u, v) for v in child c's subtree
            child_down_list.sort(key=lambda x: x[0], reverse=True)
            if child_down_list:
                child_down_max1[u] = child_down_list[0][0]
                child_down_max1_node[u] = child_down_list[0][1]
                if len(child_down_list) > 1:
                    child_down_max2[u] = child_down_list[1][0]
                # If only one child, child_down_max2 remains 0 (initialized)
            # If no children, child_down_max1/max2/node remain 0/-1 (initialized)

        # DFS2: Compute dp_up
        # This is a pre-order traversal (parent results needed for children)
        # max_path_from_p_not_via_u: max path_dist(p, v) for v NOT in subtree(u) (relative to p).
        # This value is passed down from parent p to current node u.
        def dfs2(u: int, p: int, max_path_from_p_not_via_u: int):
            # Calculate dp_up[u] = max_{v not in subtree(u)} path_dist(u, v)
            # Any path from u to v not in subtree(u) must pass through p. Path u -> p -> ... -> v.
            # path_dist(u, v) = cost_val[p] + path_dist(p, v)
            # max_{v not in subtree(u)} path_dist(u, v) = cost_val[p] + max_{v not in subtree(u)} path_dist(p, v)
            # max_{v not in subtree(u)} path_dist(p, v) is exactly the `max_path_from_p_not_via_u` passed down.
            # Base case: root (u=0), dp_up[0] = 0.
            if u != 0:
                dp_up[u] = cost_val[p] + max_path_from_p_not_via_u

            # Now, calculate the value to pass down to children v of u: `max_path_from_u_not_via_v`
            # This value represents max path_dist(u, w) for w NOT in subtree(v).
            # w can be u (path_dist(u,u)=0), or w is via parent p, or w is in sibling branch s of v (relative to u).
            # Max path_dist(u, w) for w via parent p = cost_val[p] + max path_dist(p, w') where w' is not in subtree(u) (relative to p).
            # Max over w' not in subtree(u) (relative to p) is max_path_from_p_not_via_u (passed to u).
            # So max path_dist(u, w) for w via parent p = cost_val[p] + max_path_from_p_not_via_u = dp_up[u] (if u!=0). If u=0, this part is 0.

            # Max path_dist(u, w) for w in sibling branch s of v = cost_val[s] + path_dist(s, w') where w' is in subtree(s).
            # Max over w' in subtree(s) is cost_val[s] + dp_down[s].
            # We need max over siblings s != v.

            # Find max path sum from u downwards into any child branch s (excluding s) except v.
            # Use the precomputed child_down_max1/max2/node for node u.
            # This is max (cost_val[s] + dp_down[s]) for s child of u, s != v.
            max_sibling_down_from_u = 0 # Default if u has only one child (v) or no children other than v
            if child_down_max1_node[u] != -1: # Check if u has at least one child
                 # The value cost_val[c] + dp_down[c] represents max path_dist(u, x) for x in subtree(c) via c.
                 if child_down_max1_node[u] == v:
                    # Child v was the max downward branch from u, so max sibling branch value is the second max
                    max_sibling_down_from_u = child_down_max2[u]
                 else:
                    # Child v was not the max downward branch from u, so max sibling branch value is the overall max
                    max_sibling_down_from_u = child_down_max1[u]

            # max_path_from_u_not_via_v for child v is the max of:
            # 1. Max path from u upwards (dp_up[u])
            # 2. Max path from u downwards into a sibling branch s of v (max_sibling_down_from_u)
            # The path_dist(u,u) = 0 is implicitly handled by the DP definitions representing max path *sums* which are non-negative.
            max_path_from_u_not_via_v = max(dp_up[u], max_sibling_down_from_u)

            # Recurse on children v of u, passing max_path_from_u_not_via_v
            for v in adj[u]:
                if v != p:
                    dfs2(v, u, max_path_from_u_not_via_v)


        # Run DFS1 from arbitrary root (node 0)
        dfs1(0, -1)

        # Run DFS2 from the same root (node 0)
        # For root 0, max_path_from_p_not_via_u is 0 (no parent)
        dfs2(0, -1, 0)

        # The answer for starting node i is max(dp_down[i], dp_up[i])
        # max_v path_dist(i, v)
        result = [max(dp_down[i], dp_up[i]) for i in range(n)]

        return result