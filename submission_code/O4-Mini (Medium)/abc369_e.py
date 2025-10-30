import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    # Prepare distance matrix with minimal direct bridge weights
    INF = 10**30
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
    # dist[i][j] will be shortest path from i to j
    for k in range(N):
        row_k = dist[k]
        for i in range(N):
            row_i = dist[i]
            via = row_i[k]
            if via == INF:
                continue
            # unroll j loop
            s = via
            # inline for speed
            for j in range(N):
                nd = s + row_k[j]
                if nd < row_i[j]:
                    row_i[j] = nd
    S = 0
    Tnode = N-1
    # Process queries
    Q = int(next(it))
    out = []
    for _ in range(Q):
        k = int(next(it))
        bs = [int(next(it)) - 1 for __ in range(k)]
        # build mandatory edges data
        # endpoints list length 2k
        e_nodes = [0]*(2*k)
        e_w = [0]*k
        for idx, b in enumerate(bs):
            u = U[b]
            v = V[b]
            t = T[b]
            e_nodes[2*idx] = u
            e_nodes[2*idx+1] = v
            e_w[idx] = t
        FULL = (1<<k) - 1
        # dp[mask][pos] = min cost, pos in [0..2k)
        # init with INF
        # using list comprehension
        dp = [ [INF]*(2*k) for __ in range(1<<k) ]
        # initialize from S to first edge
        for i in range(k):
            t_i = e_w[i]
            base = 1<<i
            u_idx = 2*i
            v_idx = u_idx+1
            u_node = e_nodes[u_idx]
            v_node = e_nodes[v_idx]
            # entry at u -> exit v
            c0 = dist[S][u_node] + t_i
            if c0 < dp[base][v_idx]:
                dp[base][v_idx] = c0
            # entry at v -> exit u
            c1 = dist[S][v_node] + t_i
            if c1 < dp[base][u_idx]:
                dp[base][u_idx] = c1
        # DP over masks
        for mask in range(1<<k):
            # skip empty, but may have init
            # for each current endpoint pos
            for pos in range(2*k):
                cur = dp[mask][pos]
                if cur == INF:
                    continue
                pos_node = e_nodes[pos]
                # try next edge j not in mask
                rem = FULL ^ mask
                # iterate bits in rem
                m = rem
                while m:
                    j = (m & -m).bit_length() - 1
                    m &= m-1
                    t_j = e_w[j]
                    base2 = mask | (1<<j)
                    # two orientations
                    j_u = 2*j
                    j_v = j_u+1
                    # entry at u -> exit v
                    d1 = dist[pos_node][e_nodes[j_u]] + t_j + cur
                    if d1 < dp[base2][j_v]:
                        dp[base2][j_v] = d1
                    # entry at v -> exit u
                    d2 = dist[pos_node][e_nodes[j_v]] + t_j + cur
                    if d2 < dp[base2][j_u]:
                        dp[base2][j_u] = d2
        # finalize: from full mask to Tnode
        ans = INF
        full = FULL
        for pos in range(2*k):
            c = dp[full][pos]
            if c == INF:
                continue
            val = c + dist[e_nodes[pos]][Tnode]
            if val < ans:
                ans = val
        out.append(str(ans))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()