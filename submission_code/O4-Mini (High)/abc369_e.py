import sys
import threading
def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    INF = 10**30

    # Read N, M
    N = int(next(it))
    M = int(next(it))

    # Read edges and build distance matrix
    dist = [[INF]*N for _ in range(N)]
    for i in range(N):
        dist[i][i] = 0

    U = [0]*M
    V = [0]*M
    T = [0]*M
    for i in range(M):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        t = int(next(it))
        U[i] = u
        V[i] = v
        T[i] = t
        if t < dist[u][v]:
            dist[u][v] = t
            dist[v][u] = t

    # Floyd-Warshall
    # dist[i][j] = shortest path i -> j
    N_range = range(N)
    dist_mat = dist  # local alias
    for k in N_range:
        dk = dist_mat[k]
        for i in N_range:
            di = dist_mat[i]
            via = di[k]
            if via == INF:
                continue
            s = via
            # inner loop
            # unroll local references
            for j in N_range:
                nj = s + dk[j]
                if nj < di[j]:
                    di[j] = nj

    # Answer queries
    out = []
    Q = int(next(it))
    target = N-1
    for _ in range(Q):
        K = int(next(it))
        # read required bridge indices
        Bs = [int(next(it)) - 1 for _ in range(K)]
        # build lists of endpoints and times
        a = [0]*K
        b = [0]*K
        tt = [0]*K
        for idx in range(K):
            e = Bs[idx]
            a[idx] = U[e]
            b[idx] = V[e]
            tt[idx] = T[e]

        full = (1<<K) - 1
        # dp[mask][last][side] = min cost
        # last in [0..K-1], side in {0 (at a[last]), 1 (at b[last])}
        # initialize dp with INF
        dp = [[ [INF, INF] for __ in range(K) ] for __ in range(1<<K) ]

        # base cases: mask with single bit
        # start at node 0 (island 1)
        for j in range(K):
            bit = 1<<j
            # end at a[j] (side=0): path from 1->b[j], then traverse b->a
            cost0 = dist_mat[0][b[j]] + tt[j]
            if cost0 < dp[bit][j][0]:
                dp[bit][j][0] = cost0
            # end at b[j] (side=1): path from 1->a[j], then a->b
            cost1 = dist_mat[0][a[j]] + tt[j]
            if cost1 < dp[bit][j][1]:
                dp[bit][j][1] = cost1

        # DP over masks
        for mask in range(1, full+1):
            # try transitions only if mask != single (we've done single above)
            # but our dp init covers singles; we can skip mask with only one bit if we wish
            # yet it's safe to process all masks
            for last in range(K):
                bit_last = 1<<last
                if not (mask & bit_last):
                    continue
                row = dp[mask][last]
                # if both sides INF, skip
                c0 = row[0]; c1 = row[1]
                if c0 == INF and c1 == INF:
                    continue
                # current two possible positions
                # side 0 => at a[last], side 1 => at b[last]
                # for each next edge
                for nxt in range(K):
                    bit_nxt = 1<<nxt
                    if mask & bit_nxt:
                        continue
                    nm = mask | bit_nxt
                    aj = a[nxt]; bj = b[nxt]; tj = tt[nxt]
                    # from side 0 (at a[last])
                    if c0 < INF:
                        base = c0
                        # go a[last] -> a[nxt], traverse to b[nxt], end at side=1
                        d0 = base + dist_mat[a[last]][aj] + tj
                        if d0 < dp[nm][nxt][1]:
                            dp[nm][nxt][1] = d0
                        # go a[last] -> b[nxt], traverse to a[nxt], end at side=0
                        d1 = base + dist_mat[a[last]][bj] + tj
                        if d1 < dp[nm][nxt][0]:
                            dp[nm][nxt][0] = d1
                    # from side 1 (at b[last])
                    if c1 < INF:
                        base = c1
                        # go b[last] -> a[nxt], traverse to b[nxt], end at side=1
                        d0 = base + dist_mat[b[last]][aj] + tj
                        if d0 < dp[nm][nxt][1]:
                            dp[nm][nxt][1] = d0
                        # go b[last] -> b[nxt], traverse to a[nxt], end at side=0
                        d1 = base + dist_mat[b[last]][bj] + tj
                        if d1 < dp[nm][nxt][0]:
                            dp[nm][nxt][0] = d1

        # Extract answer: full mask, then go to island N
        ans = INF
        fm = full
        for last in range(K):
            for side in (0,1):
                c = dp[fm][last][side]
                if c == INF:
                    continue
                # current node
                curr = a[last] if side == 0 else b[last]
                total = c + dist_mat[curr][target]
                if total < ans:
                    ans = total
        # Print answer
        out.append(str(ans))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()