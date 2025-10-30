import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    INF = 10**30

    # Initialize distance matrix
    dist = [ [INF]*(N+1) for _ in range(N+1) ]
    for i in range(1, N+1):
        dist[i][i] = 0

    # Read edges, store them, update dist
    U = [0]*(M+1)
    V = [0]*(M+1)
    T = [0]*(M+1)
    for i in range(1, M+1):
        u = int(next(it)); v = int(next(it)); t = int(next(it))
        U[i] = u; V[i] = v; T[i] = t
        if t < dist[u][v]:
            dist[u][v] = t
            dist[v][u] = t

    # Floyd-Warshall for all-pairs shortest paths
    for k in range(1, N+1):
        dk = dist[k]
        for i in range(1, N+1):
            di = dist[i]
            via = di[k]
            if via == INF: 
                continue
            # unroll inner
            for j in range(1, N+1):
                nd = via + dk[j]
                if nd < di[j]:
                    di[j] = nd

    Q = int(next(it))
    out = []
    # Process each query
    for _ in range(Q):
        K = int(next(it))
        bes = [int(next(it)) for __ in range(K)]
        # Prepare special edges data size K
        # We'll have 2*K states: for each edge two directions
        # start_nodes[s], end_nodes[s], and t_edge[j]
        start_nodes = [0]*(2*K)
        end_nodes   = [0]*(2*K)
        t_edge      = [0]*K
        for j in range(K):
            idx = bes[j]
            a = U[idx]; b = V[idx]; t = T[idx]
            t_edge[j] = t
            start_nodes[2*j]   = a
            end_nodes[2*j]     = b
            start_nodes[2*j+1] = b
            end_nodes[2*j+1]   = a

        FULL = (1<<K) - 1
        # dp[mask][s] = min cost to cover "mask" ending in state s
        # where state s = 2*j + d
        dp = [ [INF]*(2*K) for __ in range(1<<K) ]

        # Initialize single-edge visits
        for j in range(K):
            for d in (0,1):
                s = 2*j + d
                st = start_nodes[s]
                c = dist[1][st]
                if c < INF:
                    dp[1<<j][s] = c + t_edge[j]

        # DP over subsets
        for mask in range(1, FULL+1):
            row = dp[mask]
            # skip empty
            for s in range(2*K):
                cost_sd = row[s]
                if cost_sd >= INF:
                    continue
                j = s >> 1
                if not (mask & (1<<j)):
                    continue
                pos = end_nodes[s]
                # try to add a new edge nj
                mrem = FULL ^ mask
                # iterate bits of mrem
                mm = mrem
                while mm:
                    lb = mm & -mm
                    nj = (lb.bit_length() - 1)
                    mm ^= lb
                    base = mask | (1<<nj)
                    tnj = t_edge[nj]
                    for nd in (0,1):
                        ns = 2*nj + nd
                        stn = start_nodes[ns]
                        d2 = dist[pos][stn]
                        if d2 == INF:
                            continue
                        nc = cost_sd + d2 + tnj
                        if nc < dp[base][ns]:
                            dp[base][ns] = nc

        # finalize: full mask to N
        ans = INF
        row = dp[FULL]
        for s in range(2*K):
            c = row[s]
            if c >= INF:
                continue
            pos = end_nodes[s]
            d2 = dist[pos][N]
            if d2 < INF:
                tot = c + d2
                if tot < ans:
                    ans = tot
        out.append(str(ans))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()