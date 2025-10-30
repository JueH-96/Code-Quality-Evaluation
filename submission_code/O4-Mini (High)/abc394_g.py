import sys
from array import array
from collections import deque

def main():
    input = sys.stdin.readline
    H, W = map(int, input().split())
    N = H * W

    # Read floor counts, flatten to list F of size N
    F = [0] * N
    idx = 0
    for _ in range(H):
        row = list(map(int, input().split()))
        for v in row:
            F[idx] = v
            idx += 1

    # Build all grid edges with capacity = min(F[u], F[v])
    edges = []
    for i in range(H):
        base = i * W
        for j in range(W):
            u = base + j
            if j + 1 < W:
                v = u + 1
                fu, fv = F[u], F[v]
                edges.append((fu if fu < fv else fv, u, v))
            if i + 1 < H:
                v = u + W
                fu, fv = F[u], F[v]
                edges.append((fu if fu < fv else fv, u, v))

    # We don't need F or rows any more
    F = None

    # Sort edges by capacity descending
    edges.sort(key=lambda x: -x[0])

    # DSU for MST
    parent_dsu = list(range(N))
    rank_dsu = [0] * N

    def find(x):
        # path compression
        while parent_dsu[x] != x:
            parent_dsu[x] = parent_dsu[parent_dsu[x]]
            x = parent_dsu[x]
        return x

    def union(x, y):
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return False
        if rank_dsu[rx] < rank_dsu[ry]:
            parent_dsu[rx] = ry
        else:
            parent_dsu[ry] = rx
            if rank_dsu[rx] == rank_dsu[ry]:
                rank_dsu[rx] += 1
        return True

    # Build Maximum Spanning Tree (MST) adjacency
    adj = [[] for _ in range(N)]
    taken = 0
    for cap, u, v in edges:
        if union(u, v):
            adj[u].append((v, cap))
            adj[v].append((u, cap))
            taken += 1
            if taken == N - 1:
                break

    # Free large unused data
    edges = None
    parent_dsu = None
    rank_dsu = None

    # Precompute LCA structures on the MST to answer min-edge-on-path queries
    LOG = N.bit_length()  # enough so that 2^LOG > max depth difference

    # parent[k][v] = 2^k-th ancestor of v
    # minw[k][v] = minimum edge weight on path from v up to parent[k][v]
    parent = [array('I', [0]) * N for _ in range(LOG)]
    minw   = [array('I', [0]) * N for _ in range(LOG)]
    depth  = array('I', [0]) * N

    # BFS from node 0 to set parent[0], minw[0], depth
    dq = deque([0])
    vis = [False] * N
    vis[0] = True
    # parent[0][0] stays 0, minw[0][0] can be INF
    INF_CAP = 10**6 + 1
    minw[0][0] = INF_CAP

    while dq:
        u = dq.popleft()
        for v, w in adj[u]:
            if not vis[v]:
                vis[v] = True
                parent[0][v] = u
                minw[0][v] = w
                depth[v] = depth[u] + 1
                dq.append(v)

    # Build up the binary lifting tables
    for k in range(1, LOG):
        p_prev = parent[k - 1]
        m_prev = minw[k - 1]
        p_cur = parent[k]
        m_cur = minw[k]
        for v in range(N):
            pv = p_prev[v]
            p_cur[v] = p_prev[pv]
            # minimum on the two halves
            mw = m_prev[v]
            mw2 = m_prev[pv]
            m_cur[v] = mw if mw < mw2 else mw2

    # Process queries
    Q = int(input())
    out = []
    # bind locals for speed
    parent_loc = parent
    minw_loc = minw
    LOG_loc = LOG
    for _ in range(Q):
        A, B, Y, C, D, Z = map(int, input().split())
        u = (A - 1) * W + (B - 1)
        v = (C - 1) * W + (D - 1)
        y = Y
        z = Z

        # Lift u, v to same depth
        if depth[u] < depth[v]:
            u, v = v, u
            y, z = z, y

        diff = depth[u] - depth[v]
        mincap = INF_CAP
        # lift u by diff
        b = 0
        while diff:
            if diff & 1:
                w = minw_loc[b][u]
                if w < mincap:
                    mincap = w
                u = parent_loc[b][u]
            diff >>= 1
            b += 1

        # now same depth
        if u == v:
            bottleneck = mincap
        else:
            # lift both up until parents differ
            for k in range(LOG_loc - 1, -1, -1):
                pu = parent_loc[k][u]
                pv = parent_loc[k][v]
                if pu != pv:
                    w1 = minw_loc[k][u]
                    if w1 < mincap:
                        mincap = w1
                    w2 = minw_loc[k][v]
                    if w2 < mincap:
                        mincap = w2
                    u = pu
                    v = pv
            # one more step to reach LCA
            w1 = minw_loc[0][u]
            if w1 < mincap:
                mincap = w1
            w2 = minw_loc[0][v]
            if w2 < mincap:
                mincap = w2
            bottleneck = mincap

        # compute answer based on formula
        # if bottleneck >= min(y,z)  => cost = |y-z|
        # else => cost = y + z - 2*bottleneck
        if bottleneck >= (y if y < z else z):
            out.append(str(abs(y - z)))
        else:
            out.append(str(y + z - 2 * bottleneck))

    # Output all answers
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()