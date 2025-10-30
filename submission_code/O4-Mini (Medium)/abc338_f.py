def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    # Read edges, build dist matrix
    INF = 10**15  # large enough for sums
    dist = [ [INF]*n for _ in range(n) ]
    for i in range(n):
        dist[i][i] = 0
    for _ in range(m):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        w = int(next(it))
        if w < dist[u][v]:
            dist[u][v] = w
    # Floyd-Warshall for all-pairs shortest paths
    for k in range(n):
        dk = dist[k]
        for i in range(n):
            di = dist[i]
            tik = di[k]
            if tik == INF:
                continue
            # relax di[j] using i->k->j
            s = tik
            for j in range(n):
                vkj = dk[j]
                if vkj != INF:
                    nj = s + vkj
                    if nj < di[j]:
                        di[j] = nj
    # We'll do Held-Karp DP over subsets using a flat array of C-int to save memory
    # dp[mask, u] = min cost to cover exactly 'mask' of vertices and end at u.
    full = (1 << n) - 1
    # Quick check: if some vertex unreachable from any other, might still be OK if it's start.
    # We'll let DP find infeasible.
    if n == 0:
        print(0)
        return
    # We use array('i') but need 32-bit signed is enough (max path ~2e7)
    from array import array
    # But INF=1e9 is enough in int32
    INF32 = 10**9
    total_states = (1 << n) * n
    dp = array('i', [INF32]) * total_states
    # mask to single lowest-bit index
    mask2idx = {1 << i: i for i in range(n)}
    # initialize single-vertex masks
    for u in range(n):
        dp[(1 << u) * n + u] = 0
    # iterate through all masks
    for mask in range(1, full + 1):
        # skip singleton masks
        if mask & (mask - 1) == 0:
            continue
        base = mask * n
        # for each endpoint u in mask
        mm = mask
        # iterate bits of mask
        while mm:
            lsb = mm & -mm
            u = mask2idx[lsb]
            pmask = mask ^ lsb
            best = INF32
            # try all v in pmask
            x = pmask
            # pmask * n for indexing
            base_prev = pmask * n
            # dist col to u
            # we'll use Python int INF check, but dist[v][u] maybe large
            du = dist  # alias
            while x:
                lsb2 = x & -x
                v = mask2idx[lsb2]
                cprev = dp[base_prev + v]
                if cprev < INF32:
                    dv_u = du[v][u]
                    # dv_u may be INF large; skip if too large
                    if dv_u < INF:
                        c = cprev + dv_u
                        # check if c fits in 32-bit; but it will because bounds small
                        if c < best:
                            best = c
                x -= lsb2
            dp[base + u] = best
            mm -= lsb
    # find best over full mask
    res = INF32
    base_full = full * n
    for u in range(n):
        c = dp[base_full + u]
        if c < res:
            res = c
    if res >= INF32:
        sys.stdout.write("No")
    else:
        sys.stdout.write(str(res))

if __name__ == "__main__":
    main()