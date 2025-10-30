def main():
    import sys
    sys.setrecursionlimit(10**6)
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    mod = 998244353
    it = iter(data)
    n = int(next(it))
    N = n  # number of leaves (players are 0-indexed: 0,...,n-1)
    tot_nodes = 2 * N - 1

    # For each node (leaf or internal), we store:
    #   left_child[u] and right_child[u]: the children of u,
    #   size_arr[u]: the number of players in that team.
    left_child = [-1] * tot_nodes
    right_child = [-1] * tot_nodes
    size_arr = [0] * tot_nodes

    # Leaves represent an individual player. Their size is 1.
    for i in range(N):
        size_arr[i] = 1

    # We simulate the merge-union process with union–find.
    parent = list(range(tot_nodes))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # The tournament gives n-1 matches. We process them in order.
    # IMPORTANT: The ordering in each merge is predetermined:
    #   The team containing player p goes first (left branch) and that containing player q goes second (right branch)
    #   which determines the constant win contribution for that branch.
    next_node = N  # next available index for the new (internal) node.
    for _ in range(n - 1):
        # p and q are given 1-indexed
        p = int(next(it)) - 1
        q = int(next(it)) - 1
        rep_p = find(p)
        rep_q = find(q)
        cur = next_node
        next_node += 1
        left_child[cur] = rep_p
        right_child[cur] = rep_q
        size_arr[cur] = size_arr[rep_p] + size_arr[rep_q]
        parent[rep_p] = cur
        parent[rep_q] = cur
        parent[cur] = cur

    # The merge tree is now built; the root is the representative of any node (say, player 0).
    root = find(0)

    # Next we want to “distribute” the win probability contributions to each leaf.
    # Recall: In each match (internal node u), with left subtree size A and right subtree size B,
    #   the win probabilities for the two teams (branches) are:
    #      left branch:  A/(A+B)
    #      right branch: B/(A+B)
    # When a branch wins a match, every player in that branch gets one additional win.
    # By linearity of expectation, each leaf in the left subtree gets A/(A+B) contribution,
    # and each leaf in the right subtree gets B/(A+B).
    #
    # Once the tree is built, for every internal node u (u>=N), we want to add
    # a constant contribution to every leaf in the subtree of its left child and
    # a (possibly different) constant contribution to every leaf in the subtree of its right child.
    #
    # To do this efficiently we “flatten” the tree (only the leaves) using DFS and record
    # an Euler interval for the leaves contained in each subtree.
    
    L = [0] * tot_nodes
    R = [0] * tot_nodes
    # euler_to_player[ei] will hold the leaf's id (which is the same as its player id)
    euler_to_player = [0] * N
    cur_idx = 0
    def dfs(u):
        nonlocal cur_idx
        if left_child[u] == -1:  # this is a leaf
            L[u] = cur_idx
            R[u] = cur_idx
            euler_to_player[cur_idx] = u
            cur_idx += 1
        else:
            dfs(left_child[u])
            dfs(right_child[u])
            L[u] = L[left_child[u]]
            R[u] = R[right_child[u]]
    dfs(root)

    # We now build a Binary Indexed Tree (Fenw) for doing range-add and point-query.
    sizeBIT = N
    BIT = [0] * (sizeBIT + 1)
    def bit_update(i, val):
        i += 1
        while i <= sizeBIT:
            BIT[i] = (BIT[i] + val) % mod
            i += i & -i
    def bit_query(i):
        s = 0
        i += 1
        while i:
            s = (s + BIT[i]) % mod
            i -= i & -i
        return s % mod
    def bit_range_update(l, r, val):
        if l > r:
            return
        bit_update(l, val)
        if r + 1 < sizeBIT:
            bit_update(r + 1, (-val) % mod)

    # A small helper to compute modular inverses (mod mod).
    inv_cache = {}
    def inv(x):
        if x in inv_cache:
            return inv_cache[x]
        res = pow(x, mod - 2, mod)
        inv_cache[x] = res
        return res

    # For every internal node, update its children’s leaf intervals.
    # For node u, let A = size_arr[left_child[u]] and B = size_arr[right_child[u]].
    # Then each leaf in the left subtree gets an extra val = A/(A+B) mod mod,
    # and each leaf in the right subtree gets val = B/(A+B) mod mod.
    for u in range(N, tot_nodes):
        if left_child[u] == -1:
            continue
        tot = size_arr[u]            # tot = A+B
        inv_tot = inv(tot)
        # Left child update
        A = size_arr[left_child[u]]
        val_left = (A * inv_tot) % mod
        l_range = L[left_child[u]]
        r_range = R[left_child[u]]
        bit_range_update(l_range, r_range, val_left)
        # Right child update
        B = size_arr[right_child[u]]
        val_right = (B * inv_tot) % mod
        l_range = L[right_child[u]]
        r_range = R[right_child[u]]
        bit_range_update(l_range, r_range, val_right)

    # After processing all internal nodes, every leaf's value in the BIT equals the sum of contributions
    # from all matches on the merge path. This is exactly the expected win count for that player.
    # We then output the answers for players 1, 2, …, n in order.
    ans = [0] * N
    for euler_idx in range(N):
        player = euler_to_player[euler_idx]  # this is the leaf (player) index (0-indexed)
        res = bit_query(euler_idx) % mod
        ans[player] = res
    sys.stdout.write(" ".join(map(str, ans)))

if __name__ == '__main__':
    main()