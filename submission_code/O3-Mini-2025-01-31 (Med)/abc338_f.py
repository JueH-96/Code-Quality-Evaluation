def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    
    INF = 10**18
    # 0-indexing. Build initial distance matrix.
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for _ in range(m):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        w = int(next(it))
        dist[u][v] = w

    # Floyd–Warshall to compute best path cost from i to j.
    for k in range(n):
        for i in range(n):
            if dist[i][k] == INF:
                continue
            alt = dist[i][k]
            for j in range(n):
                nd = alt + dist[k][j]
                if nd < dist[i][j]:
                    dist[i][j] = nd

    # dp[mask] is a dictionary mapping "last vertex" -> cost.
    Nmask = 1 << n
    dp = [None] * (Nmask)
    for mask in range(Nmask):
        dp[mask] = {}
    # initialize: start from each vertex (first visit at that vertex)
    for i in range(n):
        dp[1 << i][i] = 0

    full_mask = (1 << n) - 1
    # Process states in order of increasing mask.
    # For each state represented by bitmask "mask" and ending vertex "last",
    # try to visit a new vertex u not in mask.
    for mask in range(Nmask):
        if not dp[mask]:
            continue
        # not_visited set as bits.
        not_visited = ((1 << n) - 1) ^ mask
        for last, cost_so_far in dp[mask].items():
            cand = not_visited
            while cand:
                u_bit = cand & -cand  # take lowest set bit
                u = (u_bit).bit_length() - 1
                cand -= u_bit
                if dist[last][u] == INF:
                    continue  # no route from 'last' to 'u'
                nmask = mask | (1 << u)
                new_cost = cost_so_far + dist[last][u]
                if u in dp[nmask]:
                    if new_cost < dp[nmask][u]:
                        dp[nmask][u] = new_cost
                else:
                    dp[nmask][u] = new_cost

    ans = INF
    for val in dp[full_mask].values():
        if val < ans:
            ans = val

    sys.stdout.write("No" if ans == INF else str(ans))


if __name__ == "__main__":
    main()