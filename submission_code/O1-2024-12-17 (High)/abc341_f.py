def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    edges = [tuple(map(int, input().split())) for _ in range(M)]
    W = list(map(int, input().split()))
    A = list(map(int, input().split()))

    # Build adjacency list
    adj = [[] for _ in range(N)]
    for u, v in edges:
        u -= 1
        v -= 1
        adj[u].append(v)
        adj[v].append(u)

    # Group vertices by their weights so that we can process
    # all vertices of a particular weight at once
    by_weight = [[] for _ in range(5001)]
    max_w = 0
    for i in range(N):
        w = W[i]
        by_weight[w].append(i)
        if w > max_w:
            max_w = w

    # F[x] will store the maximum number of operations obtainable
    # from a single piece starting at vertex x.
    F = [0]*N

    # We process in ascending order of w, so that for a vertex x of weight w
    # we already know F[y] for all y with W[y] < w.
    for w in range(1, max_w+1):
        # If no vertex has weight w, skip
        if not by_weight[w]:
            continue

        # If w == 1, no pieces can move to any neighbor (sum of neighbor weights < 1 is impossible),
        # so the best we can do is remove the piece there (1 operation). Hence F[x] = 1.
        if w == 1:
            for x in by_weight[w]:
                F[x] = 1
            continue

        # For each vertex x with weight w, gather neighbors with weight < w.
        # Then do a 0/1 knapsack on those neighbors: capacity = w - 1,
        # item weight = W[y], item value = F[y].
        # F[x] = 1 + max_over_knapsack_of_neighbors( sum(F[y]) ).
        for x in by_weight[w]:
            items = []
            sum_weights = 0
            sum_values = 0
            for y in adj[x]:
                if W[y] < w:
                    items.append((W[y], F[y]))
                    sum_weights += W[y]
                    sum_values += F[y]

            # Small optimization: If the sum of all neighbor weights is < w,
            # then we can choose all neighbors in one go.
            if sum_weights < w:
                F[x] = 1 + sum_values
                continue

            # Otherwise, do the classical 0/1 knapsack up to capacity = w - 1
            capacity = w - 1
            dp = [0]*(capacity + 1)  # dp[c] = best sum of F[y] with total weight c
            for wt, val in items:
                if wt <= capacity:
                    for c in range(capacity, wt - 1, -1):
                        candidate = dp[c - wt] + val
                        if candidate > dp[c]:
                            dp[c] = candidate

            F[x] = 1 + max(dp)

    # Finally, the total maximum number of operations is the sum over all vertices x
    # of A[x] * F[x], because each of the A[x] identical pieces at x can independently
    # achieve F[x] operations.
    ans = 0
    for i in range(N):
        ans += A[i] * F[i]

    print(ans)

# Do not forget to call main()!
if __name__ == "__main__":
    main()