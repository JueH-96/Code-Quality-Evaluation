def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    ptr = 0

    # Read N and M
    N = int(input_data[ptr]); ptr += 1
    M = int(input_data[ptr]); ptr += 1

    # Adjacency list (0-based)
    adj = [[] for _ in range(N)]

    # Read edges
    for _ in range(M):
        u = int(input_data[ptr]) - 1; ptr += 1
        v = int(input_data[ptr]) - 1; ptr += 1
        adj[u].append(v)
        adj[v].append(u)

    # Read W array
    W = [0]*N
    for i in range(N):
        W[i] = int(input_data[ptr])
        ptr += 1

    # Read A array
    A = [0]*N
    for i in range(N):
        A[i] = int(input_data[ptr])
        ptr += 1

    # Precompute for each vertex x: neighbors y with W[y] < W[x]
    # We'll store them in a list (cost = W[y], val = f[y]) later.
    # So first just gather the list of indices y with W[y] < W[x].
    smaller_neighbors = [[] for _ in range(N)]
    for x in range(N):
        # We will fill this after we know the order in which we compute f[y].
        # But we at least store the neighbor IDs here for later.
        # We'll filter them after we compute f for all with smaller W.
        pass

    # We will sort vertices in ascending order by W.
    # Then, when processing them in ascending order, we have f() for all strictly smaller W.
    # Ties in W do not allow picking neighbors of the same weight, since sum(W[y]) >= W[x] if W[y] = W[x].
    # So for ties in W, there is no contribution from tie-neighbors anyway.
    vertices_by_weight = sorted(range(N), key=lambda i: W[i])

    # Adjacency, but only keep neighbors y with W[y] < W[x]
    # We'll do this once, no need to do it inside the loop repeatedly.
    for x in range(N):
        smaller_neighbors[x] = []
        for y in adj[x]:
            if W[y] < W[x]:
                smaller_neighbors[x].append(y)

    # f[x] = max number of piece-removals (operations) obtainable starting with 1 piece at x
    f = [0]*N

    # We'll process x in ascending order of W[x].
    # For each x, we solve a 0-1 knapsack among y in smaller_neighbors[x] with "weight" W[y], "value" f[y],
    # capacity = W[x]-1. Then f[x] = 1 + max_{all feasible sums < W[x]} of sum of f[y].
    # If W[x] = 1, capacity = 0, so we can't pick any neighbor. Then f[x] = 1.

    idx = 0
    # We'll keep track of results in a 64-bit-like python int for safety
    # because final answer can be large (A[i] up to 1e9).
    for x in vertices_by_weight:
        cap = W[x] - 1
        if cap <= 0:
            # We can't pick any neighbors
            f[x] = 1
            continue

        # Gather items: (cost = W[y], val = f[y]) for y in smaller_neighbors[x]
        items = [(W[y], f[y]) for y in smaller_neighbors[x]]
        # 0-1 knapsack for capacity = cap
        # dp[s] = maximum sum of f[y] possible with total W[y] = s
        dp = [0]*(cap+1)
        for (cost, val) in items:
            if cost <= cap:
                # iterate backwards to do 0-1 knapsack
                for s in range(cap, cost-1, -1):
                    maybe = dp[s - cost] + val
                    if maybe > dp[s]:
                        dp[s] = maybe

        best = max(dp)
        f[x] = 1 + best

    # The maximum number of operations is sum over x of A[x] * f[x]
    # Each initial piece on x can yield f[x] total removals.
    ans = 0
    for i in range(N):
        ans += A[i] * f[i]

    print(ans)