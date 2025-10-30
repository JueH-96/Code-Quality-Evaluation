def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast index-based reading
    idx = 0
    def read_int():
        nonlocal idx
        val = int(input_data[idx])
        idx += 1
        return val

    N = read_int()
    M = read_int()

    # Adjacency list
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u = read_int() - 1
        v = read_int() - 1
        adj[u].append(v)
        adj[v].append(u)

    W = [read_int() for _ in range(N)]
    A = [read_int() for _ in range(N)]

    # We want to compute, for each vertex x, the maximum number of operations
    # f(x) a single piece starting on x can generate.  Then the final answer is
    # sum( A[x] * f(x) ) over x.  
    #
    # Because the rules say:
    #  - When removing a piece from x (that counts as 1 operation),
    #    we may place new pieces on a subset S of neighbors y with
    #    sum(W[y] for y in S) < W[x].
    #  - Each newly placed piece can then perform the same procedure, etc.
    #
    # We can show f(x) satisfies:
    #   f(x) = 1 + max_{ subset S with sum(W[y]) < W[x], y in adj[x], W[y]<W[x] }
    #                ( sum( f(y) ) ),
    # because each piece replicates from x to any chosen neighbors of strictly
    # smaller W, subject to the sum-of-weights < W[x].
    #
    # Strategy:
    #  1) Sort vertices in ascending order of W.
    #  2) For each vertex x in ascending W order:
    #       - gather neighbors y with W[y] < W[x],
    #       - solve a 0-1 knap-sack where each neighbor y is an "item" with
    #         cost = W[y], value = f(y),
    #         capacity = W[x]-1,
    #         to find the maximum sum of f(y) we can pick with total cost < W[x].
    #       - define f(x) = 1 + that maximum sum.
    #  3) Then the answer = sum_{x} A[x] * f(x).

    # Build the order of vertices by ascending W
    order = sorted(range(N), key=lambda x: W[x])

    # This will hold f(x) for each vertex
    f = [0]*N

    # Knapsack routine will be repeated for each vertex
    # but note that M <= 5000 -> total adjacency across all vertices is 2*M <=10000
    # The worst-case approach is still large, but we'll implement the standard
    # O(d(x)*W[x]) = O( sum(d(x)) * 5000 ) = 50e6 in the worst case, which is borderline
    # for Python, but we'll do it as efficiently as possible.

    # Precompute f(x) in ascending order of W
    # For the smallest W[x], f[x] = 1 since no smaller W neighbor can exist or if sum(...) < W[x] is not possible with any neighbor as W[x] might be too small.

    from math import inf

    # We'll go in ascending W.  For x, we build the items list from neighbors with strictly smaller W.
    # Then do a 1D knap-sack with capacity = W[x] - 1.
    # Finally f[x] = 1 + max dp.

    prev_W = -1
    pointer = 0

    # For quick access to f-values, we can note that once we've computed f[y],
    # we know it for when x is processed. We just do it in the sorted order.
    # We'll store (W[y], f[y]) in a list if needed.

    # We'll create an array rank_ that tells for each vertex its position in sorted order
    rank_ = [0]*N
    for r, v in enumerate(order):
        rank_[v] = r

    for r in range(N):
        x = order[r]
        w_x = W[x]
        if w_x <= 0:
            # Problem constraints say W_i >= 1, so we shouldn't see this case.
            f[x] = 1
            continue

        cap = w_x - 1
        # Gather neighbors that have W[y] < W[x]
        items = []
        for y in adj[x]:
            if W[y] < w_x:
                # f[y] is already computed because rank_[y] < rank_[x] (since W[y]<W[x])
                items.append((W[y], f[y]))

        if len(items) == 0 or cap < 0:
            # No smaller neighbor or no capacity
            f[x] = 1
            continue

        # Sort items by cost ascending
        items.sort(key=lambda z: z[0])

        # knap-sack in 1D dp up to capacity cap
        # dp[c] = maximum sum of f(y) with total cost c
        # We'll track only c up to cap
        dp = [ -1]*(cap+1)
        dp[0] = 0
        # do the standard 0-1 knap-sack
        for cost, val in items:
            if cost > cap:
                # can't fit at all
                continue
            # iterate backwards
            for c in range(cap, cost-1, -1):
                if dp[c - cost] >= 0:
                    new_val = dp[c - cost] + val
                    if new_val > dp[c]:
                        dp[c] = new_val

        best_val = max(dp)
        f[x] = 1 + best_val

    # Now compute the final answer = sum(A[x]*f[x])
    # Watch out for large values, use a 64-bit integer
    ans = 0
    for i in range(N):
        ans += A[i] * f[i]

    print(ans)