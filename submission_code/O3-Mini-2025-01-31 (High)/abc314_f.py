def main():
    import sys
    sys.setrecursionlimit(10**6)
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    mod = 998244353
    it = iter(data)
    n = int(next(it))
    
    # We'll build a merge tree as described.
    # Indices 1..n are the initial leaves (players).
    # For each match i (1-indexed), we create a new node with id = n + i.
    # That new node has two children:
    #   - left: team that contained player p_i (the first team)
    #   - right: team that contained player q_i (the second team)
    # and we also record the sizes of the two teams (which are fixed regardless of match outcome)
    #
    # Each match gives, for every player in the winning team:
    #   +1 win if that team wins.
    # And the winning probability is:
    #   first team: (a)/(a+b)
    #   second team: (b)/(a+b)
    # Thus, in expectation each player in the first team gets an extra a/(a+b) 
    # and each in the second team gets extra b/(a+b).
    #
    # Since expectation is linear, if we build the merge tree,
    # the expected win for a leaf (player) is the sum over all matches 
    # in whose corresponding merge node the leaf is in the team that gets the win.
    # That is, along the unique tree path from the leaf to the root,
    # at an internal node where the node has two children with sizes L and R,
    # if the leaf is in the left child then add L/(L+R),
    # if it is in the right child then add R/(L+R).
    #
    # We compute these values in modulo arithmetic.
    
    # We'll simulate the tournament using a union‐find structure.
    # We need arrays for parent pointers and sizes; note that the final node indices go up to (n + n - 1).
    N_tot = 2 * n  # allocate index range [0, 2*n]
    parent = list(range(N_tot + 1))
    size_arr = [0] * (N_tot + 1)
    # For leaves (players 1..n), size=1
    for i in range(1, n+1):
        size_arr[i] = 1
    # tree array: for internal nodes (id from n+1 to n+(n-1)), store tuple (left, right, left_size, right_size).
    tree = [None] * (N_tot + 1)
    
    # We'll implement an iterative union-find "find".
    def uf_find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    # Process the N-1 matches.
    # Match i (i=1...n-1) uses p_i, q_i.
    for i in range(1, n):
        p = int(next(it))
        q = int(next(it))
        u = uf_find(p)
        v = uf_find(q)
        new_id = n + i  # new internal node id
        left_size = size_arr[u]
        right_size = size_arr[v]
        tree[new_id] = (u, v, left_size, right_size)
        size_arr[new_id] = left_size + right_size
        # Attach the two merged teams
        parent[u] = new_id
        parent[v] = new_id
        parent[new_id] = new_id
    
    # Final merge tree root is the representative of any leaf (say, player 1)
    final_root = uf_find(1)
    
    # Precompute modular inverses for numbers 1 .. n.
    # Note: the denominators in our probabilities are a+b which are between 2 and n.
    inv = [0] * (n + 1)
    inv[1] = 1
    for i in range(2, n+1):
        inv[i] = (mod - (mod // i) * inv[mod % i] % mod) % mod

    # Now do a DFS from the root.
    # For each internal node, if the node (match) has children with sizes ls and rs,
    # then if we go into the left child we add (ls/(ls+rs)) mod mod,
    # and if we go to the right child we add (rs/(ls+rs)) mod mod.
    # The expected win for a leaf (player) is the sum of these contributions along its root path.
    # We do an iterative DFS using an explicit stack.
    res = [0] * (n + 1)   # answer for players 1..n
    stack = [(final_root, 0)]
    while stack:
        node, acc = stack.pop()
        if node <= n:
            # Leaf: record the accumulated expected wins.
            res[node] = acc % mod
        else:
            # Internal match node. Retrieve stored info.
            left, right, ls, rs = tree[node]
            tot = ls + rs
            new_val_left = (ls * inv[tot]) % mod  # ls/(ls+rs)
            new_val_right = (rs * inv[tot]) % mod # rs/(ls+rs)
            stack.append((left, (acc + new_val_left) % mod))
            stack.append((right, (acc + new_val_right) % mod))
    
    # Prepare output in order for players 1..n.
    out = " ".join(str(res[i] % mod) for i in range(1, n+1))
    sys.stdout.write(out)
    
if __name__ == '__main__':
    main()