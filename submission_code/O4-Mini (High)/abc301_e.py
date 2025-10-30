import sys
def main():
    import sys
    from collections import deque
    import array

    data = sys.stdin.read().split()
    it = iter(data)
    try:
        H = int(next(it))
    except StopIteration:
        return
    W = int(next(it))
    T = int(next(it))
    raw = [next(it) for _ in range(H)]
    # find S, G, candies
    candies = []
    si = sj = gi = gj = -1
    for i in range(H):
        row = raw[i]
        for j in range(W):
            c = row[j]
            if c == 'S':
                si, sj = i, j
            elif c == 'G':
                gi, gj = i, j
            elif c == 'o':
                candies.append((i, j))
    # free grid
    free = [[ch != '#' for ch in row] for row in raw]

    # nodes: 0 = S, 1..N = candies, N+1 = G
    nodes = [(si, sj)] + candies + [(gi, gj)]
    M = len(nodes)
    N = M - 2  # number of candies

    # map grid positions to node index, -1 if none
    Hvar = H; Wvar = W
    node_id = [[-1] * Wvar for _ in range(Hvar)]
    for idx, (x, y) in enumerate(nodes):
        node_id[x][y] = idx

    # compute pairwise distances with BFS from S and each candy (skip G)
    INF_large = 10**9
    dist = [[INF_large] * M for _ in range(M)]
    free_local = free
    node_id_local = node_id
    for u in range(M - 1):  # 0..N
        sx, sy = nodes[u]
        # BFS
        d2 = [[-1] * Wvar for _ in range(Hvar)]
        dq = deque()
        d2[sx][sy] = 0
        dq.append((sx, sy))
        dist_row = dist[u]
        found = 0
        target = M  # number of nodes to find
        while dq:
            i, j = dq.popleft()
            du = d2[i][j]
            v = node_id_local[i][j]
            if v != -1:
                dist_row[v] = du
                found += 1
                if found == target:
                    break
            # expand neighbors
            ni = i - 1
            if ni >= 0:
                if d2[ni][j] < 0 and free_local[ni][j]:
                    d2[ni][j] = du + 1
                    dq.append((ni, j))
            ni = i + 1
            if ni < Hvar:
                if d2[ni][j] < 0 and free_local[ni][j]:
                    d2[ni][j] = du + 1
                    dq.append((ni, j))
            nj = j - 1
            if nj >= 0:
                if d2[i][nj] < 0 and free_local[i][nj]:
                    d2[i][nj] = du + 1
                    dq.append((i, nj))
            nj = j + 1
            if nj < Wvar:
                if d2[i][nj] < 0 and free_local[i][nj]:
                    d2[i][nj] = du + 1
                    dq.append((i, nj))
    # extract needed distances
    # dist from S to G
    dsg = dist[0][N+1]
    # dist from S to each candy
    ds = [dist[0][i+1] for i in range(N)]
    # dist between candies
    cdist = [[dist[i+1][j+1] for j in range(N)] for i in range(N)]
    # dist from each candy to G
    dg = [dist[i+1][N+1] for i in range(N)]

    # DP over subsets of candies
    INFdp = T + 1
    if N > 0:
        total = (1 << N) * N
        dp = array.array('I', [INFdp]) * total
        # initialize single-candy masks
        for i in range(N):
            di = ds[i]
            if di < INF_large:
                dp[(1 << i) * N + i] = di
        # flatten references
        dp_local = dp
        cdist_local = cdist
        N_local = N
        INF_local = INFdp
        # DP transitions
        for mask in range(1, 1 << N_local):
            # skip single-bit masks
            if mask & (mask - 1) == 0:
                continue
            base = mask * N_local
            m_i = mask
            while m_i:
                lb_i = m_i & -m_i
                i = lb_i.bit_length() - 1
                prev = mask ^ lb_i
                base_prev = prev * N_local
                best = INF_local
                m_j = prev
                # try last candy j
                while m_j:
                    lb_j = m_j & -m_j
                    j = lb_j.bit_length() - 1
                    cost = dp_local[base_prev + j] + cdist_local[j][i]
                    if cost < best:
                        best = cost
                    m_j ^= lb_j
                dp_local[base + i] = best
                m_i ^= lb_i

    # compute answer
    ans = -1
    # no-candy direct path?
    if dsg <= T:
        ans = 0
    if N > 0:
        dp_local = dp
        dg_local = dg
        N_local = N
        # check subsets
        for mask in range(1, 1 << N_local):
            bc = mask.bit_count()
            if bc <= ans:
                continue
            base = mask * N_local
            m_i = mask
            ok = False
            while m_i:
                lb_i = m_i & -m_i
                i = lb_i.bit_length() - 1
                if dp_local[base + i] + dg_local[i] <= T:
                    ans = bc
                    break
                m_i ^= lb_i

    print(ans)

if __name__ == "__main__":
    main()