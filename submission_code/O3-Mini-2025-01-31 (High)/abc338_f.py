def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    try:
        n = int(next(it))
        m = int(next(it))
    except StopIteration:
        return

    # Set a large INF. (Edge weights up to 10^6 and at most 20 edges in a cover‐path.)
    INF = 10**15
    # Build the initial distance matrix.
    d = [[INF] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0
    for _ in range(m):
        u = int(next(it)) - 1  # 0-index vertices.
        v = int(next(it)) - 1
        w = int(next(it))
        d[u][v] = w

    # Run Floyd–Warshall to compute d[u][v] (shortest cost from u to v)
    for k in range(n):
        dk = d[k]
        for i in range(n):
            di = d[i]
            ik = di[k]
            if ik == INF:
                continue
            for j in range(n):
                new_val = ik + dk[j]
                if new_val < di[j]:
                    di[j] = new_val

    # The idea is that any valid walk “first–visits” vertices in some order.
    # Hence, the optimum cost equals the minimum “Hamiltonian path” cost on the complete
    # graph with cost d[u][v]. We now compute this by DP.
    #
    # We'll use a level–DP (building states by number r of vertices visited) where for
    # each state (mask, last) we keep the best cost. Here "mask" is a bit–mask that represents
    # the set of visited vertices, and "last" is the vertex where the current (first–visit) walk ends.
    #
    # dp[r] will be a dictionary: key = mask (with exactly r set bits), value = dictionary mapping
    # an ending vertex (last) to the best cost.
    #
    # Initialize level 1 (one–vertex states).
    full_mask = (1 << n) - 1
    bits = [1 << i for i in range(n)]
    dp = [ {} for _ in range(n+1) ]
    for i in range(n):
        dp[1][bits[i]] = { i: 0 }

    # Now, for r=1,...,n-1, extend each state by adding a vertex not in the mask.
    # We use the fact that for each state (mask, v) the next possible vertex w must belong
    # to the "complement" (full_mask ^ mask).
    for r in range(1, n):
        cur_level = dp[r]
        next_level = dp[r+1]
        for mask, endDict in cur_level.items():
            # comp: bitmask representing vertices not yet visited.
            comp = full_mask ^ mask
            for v, curCost in endDict.items():
                temp = comp
                while temp:
                    lb = temp & -temp  # lowest set bit
                    w = lb.bit_length() - 1  # vertex corresponding to that bit
                    temp -= lb
                    # Use d[v][w] (which might even use a “detour” that visits already visited vertices)
                    if d[v][w] == INF:
                        continue
                    new_mask = mask | lb
                    new_cost = curCost + d[v][w]
                    subd = next_level.get(new_mask)
                    if subd is None:
                        next_level[new_mask] = { w: new_cost }
                    else:
                        prev = subd.get(w, INF)
                        if new_cost < prev:
                            subd[w] = new_cost

    # The answer is the minimum cost among states that visited all vertices.
    ans = INF
    final = dp[n].get(full_mask)
    if final is not None:
        for cost in final.values():
            if cost < ans:
                ans = cost

    out = sys.stdout
    if ans == INF:
        out.write("No")
    else:
        out.write(str(ans))

if __name__ == '__main__':
    main()