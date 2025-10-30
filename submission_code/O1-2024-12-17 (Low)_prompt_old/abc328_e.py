def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M, K = map(int, input_data[:3])
    edges = input_data[3:]
    
    # Adjacency matrix: -1 if no edge, else w mod K
    # We'll store w already mod K to save time later.
    adj = [[-1]*N for _ in range(N)]
    idx = 0
    for _ in range(M):
        u = int(edges[idx]); v = int(edges[idx+1]); w = int(edges[idx+2])
        idx += 3
        u -= 1
        v -= 1
        w_mod = w % K
        adj[u][v] = w_mod
        adj[v][u] = w_mod

    # If N == 1 (should not happen by constraints), just print 0
    # If N == 2, there's exactly one spanning tree (the single edge), so just output that.
    if N == 2:
        # The graph has exactly one edge or possibly more, but any spanning tree of 2 vertices is just one edge connecting them.
        # By constraints, the graph is connected and simple, so there must be an edge (0,1).
        print(adj[0][1] % K)
        return

    # For N >= 3:
    #
    # We'll enumerate all possible Prufer codes of length N-2, each "symbol" in [0..N-1].
    # Number of such codes = N^(N-2), which for N<=8 is at most 8^6 = 262144 (feasible in Python).
    #
    # Reconstruct the corresponding spanning tree (if possible) and compute sum of weights mod K.
    # Keep track of the minimal modulo value.

    # Fast exponent / range for N^(N-2).  Special case: N^(N-2) = 1 if N=2 (handled above) or if N-2=0.
    # In Python, pow(N, N-2) works, but handle N=2 separately. For N>=3, just do it directly.
    count_codes = pow(N, N-2)

    min_cost = K-1  # Since cost mod K is in [0..K-1], start from the highest possible mod value

    # A small helper to insert x into sorted list 'lst' (size at most N) quickly.
    # Because N <= 8, a simple linear insertion is fine.
    def insert_sorted(lst, x):
        # We keep it sorted in ascending order
        i = 0
        while i < len(lst) and lst[i] < x:
            i += 1
        lst.insert(i, x)

    for code_int in range(count_codes):
        # Decode code_int into a Prufer code p of length N-2 in base N
        p = []
        tmp = code_int
        for _ in range(N-2):
            p.append(tmp % N)
            tmp //= N

        # Compute degrees: each vertex starts with degree=1, plus 1 for each time it appears in the code
        degrees = [1]*N
        for x in p:
            degrees[x] += 1

        # Make a sorted list of leaves (all vertices with degree=1)
        leaves = [i for i in range(N) if degrees[i] == 1]
        leaves.sort()

        skip = False
        total = 0  # We'll keep the sum mod K in this variable

        # Reconstruct edges from the Prufer code
        for x in p:
            if not leaves:  
                # should never happen if the code is valid, but just in case
                skip = True
                break
            u = leaves[0]
            # remove u from the front
            del leaves[0]
            # The edge (u, x) must exist
            if adj[u][x] < 0:
                skip = True
                break
            # add its weight
            total = (total + adj[u][x]) % K
            # update degrees
            degrees[u] -= 1
            degrees[x] -= 1
            # if x becomes a leaf now, insert it
            if degrees[x] == 1:
                insert_sorted(leaves, x)

            if degrees[u] < 0 or degrees[x] < 0:
                skip = True
                break
            # continue

        if skip:
            continue

        # Now exactly 2 vertices remain as leaves
        if len(leaves) != 2:
            continue
        a, b = leaves[0], leaves[1]
        if adj[a][b] < 0:
            continue

        total = (total + adj[a][b]) % K

        if total < min_cost:
            min_cost = total
            # If we ever find 0, we can't do better, so break early
            if min_cost == 0:
                break

    print(min_cost)