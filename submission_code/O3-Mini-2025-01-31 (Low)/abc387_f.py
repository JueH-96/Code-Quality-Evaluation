def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = [0] * (N + 1)
    for i in range(1, N + 1):
        A[i] = int(next(it))
    mod = 998244353

    # In our problem every node i gets an edge i -> A[i]. Hence the “dependency” is:
    #    x_i <= x_{A[i]}
    # Notice that because every node points to one other, the directed graph consists of a number of
    # components each having exactly one cycle (which might be a self–loop). In any cycle, by chaining
    # inequalities we deduce that all nodes must receive the same value.
    #
    # Outside the cycles there are trees whose roots are in the cycle. When a tree node u has parent v,
    # the condition is x_u <= x_v. If v gets a value v0, then u can take any value from 1 up to v0 (subject
    # to the same restrictions in its subtree). Thus the choices in each tree “below” a fixed node become
    # independent.
    #
    # Our plan is:
    # 1. Identify the cycle nodes. (Mark every node that belongs to a cycle.)
    # 2. For any edge from a cycle node (or any node) to another node that is not in the cycle – we view that edge
    #    as the root of a tree branch.
    # 3. For tree branches we define for each node u a DP array dp[u] of length M+1 where:
    #       dp[u][p] = number of ways to assign values for the subtree rooted at u,
    #         under the assumption that the maximum allowed value for u is p.
    #    Here the recurrence is as follows. For a node u (which is not in a cycle) with children (in the tree)
    #    c1, c2, ... , ck (each child attached via an edge u -> child that is NOT a cycle edge), if we “fix”
    #    the value for u to be some v (1 <= v <= p) then each child c must get an assignment from 1..v.
    #    But if we define (for fixed u) the “exact‐value” contribution as
    #        f(u, v) = ∏ ( dp[c][v] - dp[c][v-1] )   for each tree child c  (with dp[child][0] = 0)
    #    then we have:
    #        dp[u][p] = sum_{v = 1}^{p} f(u, v).
    #    For a leaf (with no tree children) we have f(u, v)=1 and so dp[u][p] = p.
    # 
    # 4. For a cycle node u, it will appear as the “root” of a dangling tree (the branch going out from the cycle).
    #    We need only the “exact‐value” contribution, namely:
    #         f_cycle(u, v) = ∏ ( dp[c][v] - dp[c][v-1] ),  where c are tree children of u.
    #    (If u has no tree children, f_cycle(u, v) is defined as 1 for every v.)
    #
    # 5. Finally, for a cycle (say with nodes in the list comp) all nodes must get the same value v.
    #    And the contribution of the entire cycle is:
    #         Sum_{v=1}^{M}  ∏_{u in comp} f_cycle(u, v)
    #    and the answer is the product (modulo mod) of the contributions of every cycle component.
    #
    # --------------------------------------------
    # Step 1. Identify cycle nodes.
    #
    # We follow the chain from every node (since there is 1 out‐edge per node) and use a “coloring” (or
    # visitation state) to mark cycles.
    in_cycle = [False] * (N + 1)
    visited = [0] * (N + 1)  # 0 = unvisited, 1 = visiting, 2 = finished

    def find_cycle(u):
        stack = []
        cur = u
        while True:
            if visited[cur] == 0:
                visited[cur] = 1
                stack.append(cur)
                cur = A[cur]
                continue
            elif visited[cur] == 1:
                # Found a cycle; nodes from first occurrence of cur in stack are in the cycle.
                cycle_start = stack.index(cur)
                for node in stack[cycle_start:]:
                    in_cycle[node] = True
                for node in stack:
                    visited[node] = 2
                break
            else:
                # Already finished, no new cycle here.
                for node in stack:
                    visited[node] = 2
                break

    for i in range(1, N + 1):
        if visited[i] == 0:
            find_cycle(i)

    # Build the tree structure.
    # For each node v, consider all incoming edges from u with A[u] == v.
    # We want to “separate” tree edges from cycle edges.
    tree_children = [[] for _ in range(N + 1)]
    for v in range(1, N + 1):
        # For every edge from u -> v
        # If u is not in a cycle then u is in a tree attached to v.
        # (Even if v is in a cycle, the branch u is off the cycle.)
        # Note: if both u and v are in a cycle, then this edge is part of the cycle and we do not treat it as a tree edge.
        # (Also, a self–loop (u == v) is a cycle.)
        for u in [] if v > N else []:
            pass  # dummy, not used
        for u in []:  # dummy placeholder
            pass
    # We can build tree_children easily by iterating over nodes from 1..N:
    for u in range(1, N + 1):
        if not in_cycle[u]:
            par = A[u]
            tree_children[par].append(u)
            
    # Step 2. Compute dp for tree nodes.
    # dp[u][p] = number of assignments for subtree rooted at u given that the maximum allowed value for u is p.
    #
    # We use a DFS (post–order) on the tree branches.
    dp_tree = {}  # memoization: dp_tree[u] is a list of length M+1, indices 0..M.
    def dfs_tree(u):
        if u in dp_tree:
            return dp_tree[u]
        dp = [0] * (M + 1)
        # For each tree child c (those attached to u, i.e. not in cycle—they come from tree_children[u]):
        child_dp_list = []
        for c in tree_children[u]:
            child_dp_list.append(dfs_tree(c))
        # For each allowed value x (from 1 to M) this node u might be assigned,
        # the “exact” contribution is the product over children:
        #   f(u, x) = ∏_{child c} (dp[c][x] - dp[c][x-1])
        # (For a leaf u, the product over an empty set is 1.)
        # Then dp[u][p] = sum_{x=1}^{p} f(u, x)
        fvals = [0] * (M + 1)  # fvals[x] for x in 1..M; index 0 unused.
        for x in range(1, M + 1):
            prod = 1
            for child_dp in child_dp_list:
                # For convenience define dp[c][0] = 0.
                diff = (child_dp[x] - child_dp[x - 1]) % mod
                prod = (prod * diff) % mod
            fvals[x] = prod
        for p in range(1, M + 1):
            dp[p] = (dp[p - 1] + fvals[p]) % mod
        dp_tree[u] = dp
        return dp

    # For cycle nodes, we do not “choose” their value by DP (because
    # all nodes in a cycle must have the same assignment) but we do need
    # the contributions from their hanging tree branches.
    # Define for a cycle node u:
    #    f_cycle(u, v) = ∏_{child in tree_children[u]} (dp[child][v] - dp[child][v-1])
    # (if u has no tree children, f_cycle(u, v)=1 for each v.)
    f_cycle = [None] * (N + 1)
    def compute_f_cycle(u):
        if f_cycle[u] is not None:
            return f_cycle[u]
        res = [0] * (M + 1)  # for v=0..M; we use indices 1..M.
        for v in range(1, M + 1):
            prod = 1
            for c in tree_children[u]:
                child_dp = dfs_tree(c)
                diff = (child_dp[v] - child_dp[v - 1]) % mod
                prod = (prod * diff) % mod
            res[v] = prod
        f_cycle[u] = res
        return res

    # Step 3. Identify the cycle components.
    # For each node that is part of a cycle (in_cycle[u]==True), we “follow” the cycle’s links.
    processed = [False] * (N + 1)
    cycles = []  # list of lists; each inner list is a cycle (listing its nodes)
    for i in range(1, N + 1):
        if in_cycle[i] and not processed[i]:
            comp = []
            cur = i
            while True:
                comp.append(cur)
                processed[cur] = True
                cur = A[cur]
                if cur == i:
                    break
            cycles.append(comp)

    # Step 4. For each cycle, the nodes in the cycle must all get the same value v.
    # For a cycle component comp, the contribution is:
    #    Sum_{v=1}^{M}  (∏_{u in comp} f_cycle(u, v) )
    # The overall answer is the product (mod mod) over all cycles.
    ans = 1
    for comp in cycles:
        comp_sum = 0
        for v in range(1, M + 1):
            prod = 1
            for u in comp:
                f_arr = compute_f_cycle(u)
                prod = (prod * f_arr[v]) % mod
            comp_sum = (comp_sum + prod) % mod
        ans = (ans * comp_sum) % mod

    sys.stdout.write(str(ans))


if __name__ == '__main__':
    main()