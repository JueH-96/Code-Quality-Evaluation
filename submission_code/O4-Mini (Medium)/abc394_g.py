import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    from collections import deque

    data = sys.stdin.readline
    H, W = map(int, data().split())
    N = H * W
    F = [0] * N
    for i in range(H):
        row = list(map(int, data().split()))
        base = i * W
        for j in range(W):
            F[base + j] = row[j]

    # Build edge list: for each cell, edges to right and down
    edges = []
    # pre-allocate to roughly 2*N
    # We'll collect (weight, u, v)
    for i in range(H):
        for j in range(W):
            u = i * W + j
            f_u = F[u]
            # right
            if j + 1 < W:
                v = u + 1
                w = f_u if f_u < F[v] else F[v]
                edges.append((w, u, v))
            # down
            if i + 1 < H:
                v = u + W
                w = f_u if f_u < F[v] else F[v]
                edges.append((w, u, v))

    # sort edges by descending weight
    edges.sort(key=lambda x: x[0], reverse=True)

    # DSU to build Maximum Spanning Tree (by edge weights descending)
    parent = list(range(N))
    size = [1] * N
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        return True

    # adjacency list for MST
    adj = [[] for _ in range(N)]
    edge_count = 0
    for w, u, v in edges:
        if union(u, v):
            adj[u].append((v, w))
            adj[v].append((u, w))
            edge_count += 1
            if edge_count == N - 1:
                break
    # done MST

    # Prepare LCA with binary lifting, track min edge on path
    # K levels: need up to log2(N) ~ 18, use 19 for safety
    K = 19
    INF = 10**9
    up = [ [0] * N for _ in range(K) ]
    medg = [ [INF] * N for _ in range(K) ]
    depth = [ -1 ] * N

    # BFS from node 0 as root
    dq = deque()
    root = 0
    depth[root] = 0
    up[0][root] = root
    medg[0][root] = INF
    dq.append(root)
    while dq:
        u = dq.popleft()
        for v, w in adj[u]:
            if depth[v] == -1:
                depth[v] = depth[u] + 1
                up[0][v] = u
                medg[0][v] = w
                dq.append(v)

    # build ancestors
    for k in range(1, K):
        up_k = up[k]
        up_k1 = up[k-1]
        up_k1_v = up[k-1]
        medg_k = medg[k]
        medg_k1 = medg[k-1]
        medg_k1_up = medg[k-1]
        for v in range(N):
            p = up_k1[v]
            up_k[v] = up_k1[p]
            # min edge to 2^k ancestor is min of two halves
            a = medg_k1[v]
            b = medg_k1[p]
            medg_k[v] = a if a < b else b

    # function to get minimum edge weight on path u-v
    def query_min_edge(u, v):
        # ensures u is deeper
        if depth[u] < depth[v]:
            u, v = v, u
        res = INF
        dd = depth[u] - depth[v]
        # lift u up by dd
        bit = 0
        while dd:
            if dd & 1:
                w = medg[bit][u]
                if w < res: res = w
                u = up[bit][u]
            dd >>= 1
            bit += 1
        if u == v:
            return res
        # lift both together
        for k in range(K-1, -1, -1):
            if up[k][u] != up[k][v]:
                w1 = medg[k][u]
                if w1 < res: res = w1
                w2 = medg[k][v]
                if w2 < res: res = w2
                u = up[k][u]
                v = up[k][v]
        # now parents are same = LCA
        # include edge to parent
        w1 = medg[0][u]
        if w1 < res: res = w1
        w2 = medg[0][v]
        if w2 < res: res = w2
        return res

    # Process queries
    out = []
    Q = int(data())
    for _ in range(Q):
        Ai, Bi, Yi, Ci, Di, Zi = map(int, data().split())
        # zero-based node ids
        u = (Ai-1) * W + (Bi-1)
        v = (Ci-1) * W + (Di-1)
        M = query_min_edge(u, v)
        # now compute answer
        # if min(Yi,Zi) <= M: ans = abs(Yi - Zi)
        # else: ans = Yi + Zi - 2*M
        if (Yi if Yi < Zi else Zi) <= M:
            ans = Yi - Zi if Yi > Zi else Zi - Yi
        else:
            ans = Yi + Zi - 2*M
        out.append(str(ans))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()