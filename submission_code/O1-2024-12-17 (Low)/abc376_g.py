def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    MOD = 998244353

    # Fast exponentiation to find modular inverse (Fermat's little theorem).
    def modinv(a, m=MOD):
        return pow(a, m - 2, m)

    t = int(input_data[0])
    idx_data = 1

    # We will process each test case
    outputs = []
    for _ in range(t):
        N = int(input_data[idx_data]); idx_data += 1
        parents = list(map(int, input_data[idx_data : idx_data + N]))
        idx_data += N
        a = list(map(int, input_data[idx_data : idx_data + N]))
        idx_data += N

        # Build adjacency list (children) for the tree
        # We have vertices 0..N, root=0.  p_i is parent of i (i=1..N).
        children = [[] for _ in range(N+1)]
        for i in range(1, N+1):
            p = parents[i-1]
            children[p].append(i)

        # Subtree sum array: f[u] = sum of a-values in subtree of u.
        # For convenience, define a_0 = 0 (since vertex 0 has no treasure probability).
        f = [0]*(N+1)
        # a_i (i=1..N) is stored in a[i-1].  So f(i) = a[i-1] plus children's sums, for i>=1.
        # for i=0, a_0=0

        def dfs_subtree_sum(u):
            """Compute f[u] = sum of a_i over subtree of u."""
            # If u != 0, add the weight a[u-1], otherwise 0
            s = 0 if u == 0 else a[u-1]
            for c in children[u]:
                s += dfs_subtree_sum(c)
            f[u] = s
            return s

        dfs_subtree_sum(0)  # compute all subtree sums

        # Now we build a DFS ordering of the nodes (excluding 0),
        # always visiting children in descending order of f[child].
        # The position of node i in this ordering (1-based) is the cost
        # (the step at which we search node i).  The expected cost is
        #   ( sum_{i=1..N}( a_i * position(i) ) ) / ( sum_{i=1..N} a_i ).

        order_pos = [0]*(N+1)  # will store the 1-based position for each node
        current_index = 0      # global counter for positions in the DFS order

        def build_order(u):
            nonlocal current_index
            # If u != 0, add it to the order
            if u != 0:
                current_index += 1
                order_pos[u] = current_index
            # Sort children by descending f-value
            if children[u]:
                children[u].sort(key=lambda c: f[c], reverse=True)
                for c in children[u]:
                    build_order(c)

        build_order(0)

        # total_a = sum(a_i) for i=1..N
        total_a = sum(a)
        inv_total_a = modinv(total_a % MOD)

        # Compute sum_{i=1..N} a_i * pos(i)  mod M
        # note that i in code is 1..N (the vertex label),
        # and a[i-1] is the weight of vertex i.
        # order_pos[i] is the 1-based position of i in the search order.
        s = 0
        for i in range(1, N+1):
            s += a[i-1] * order_pos[i]
        s_mod = s % MOD

        # Expected cost = s_mod * inv_total_a (mod M)
        ans = (s_mod * inv_total_a) % MOD
        outputs.append(str(ans))

    print("
".join(outputs))