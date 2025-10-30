def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    # Build graph as an adjacency list.
    # Every edge goes from a smaller-index vertex to a larger
    # one (u < v) so the natural order 1..n is a valid topological order.
    graph = [[] for _ in range(n + 1)]
    max_b = 0
    min_c = 10**9
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        b = int(next(it))
        c = int(next(it))
        graph[u].append((v, b, c))
        if b > max_b:
            max_b = b
        if c < min_c:
            min_c = c

    # We know that the maximum ratio cannot exceed max(b)/min(c)
    lo = 0.0
    hi = max_b / min_c  # hi is an upper bound on the maximum ratio

    # In the DAG with positive cost along any path,
    # define for any candidate R the value:
    #    dp[v] = maximum total (beauty - R * cost) along a path from 1 to v.
    # Then for a given R, there exists a path with ratio at least R if and only if dp[n] >= 0.
    NEG_INF = float('-inf')

    def possible(R):
        dp = [NEG_INF] * (n + 1)
        dp[1] = 0.0
        # Process vertices in the natural order; since for every edge u -> v, u < v,
        # we are sure dp[u] is already computed.
        for u in range(1, n + 1):
            dpu = dp[u]
            if dpu == NEG_INF:
                continue
            for v, b, c in graph[u]:
                cand = dpu + b - R * c
                if cand > dp[v]:
                    dp[v] = cand
        return dp[n] >= 0.0

    # Binary search for the maximum R such that some path from 1 to n has
    # total beauty - R * total cost >= 0, i.e. (sum(b)/sum(c)) >= R.
    # Because dp(n) as a function of R is monotonic (decreasing in R),
    # we can binary search for the maximum achievable ratio.
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if possible(mid):
            lo = mid
        else:
            hi = mid

    # Print the answer with 16 decimals; error <= 1e-9 is acceptable.
    sys.stdout.write("{:.16f}".format(lo))

if __name__ == '__main__':
    main()