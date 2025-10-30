import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    H, W = map(int, input().split())
    F = [list(map(int, input().split())) for _ in range(H)]
    N = H * W

    # map (i,j) to node id 0..N-1
    def node(i, j):
        return i * W + j

    # Build all edges with weight = min(F[u], F[v])
    edges = []
    for i in range(H):
        for j in range(W):
            u = node(i, j)
            if i + 1 < H:
                v = node(i+1, j)
                w = F[i][j] if F[i][j] < F[i+1][j] else F[i+1][j]
                edges.append((w, u, v))
            if j + 1 < W:
                v = node(i, j+1)
                w = F[i][j] if F[i][j] < F[i][j+1] else F[i][j+1]
                edges.append((w, u, v))

    # Kruskal for maximum spanning forest
    parentUF = list(range(N))
    rankUF = [0]*N
    def find(x):
        while parentUF[x] != x:
            parentUF[x] = parentUF[parentUF[x]]
            x = parentUF[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rankUF[rx] < rankUF[ry]:
            parentUF[rx] = ry
        else:
            parentUF[ry] = rx
            if rankUF[rx] == rankUF[ry]:
                rankUF[rx] += 1
        return True

    # sort edges desc by w
    edges.sort(key=lambda x: x[0], reverse=True)

    # adjacency for MST
    adj = [[] for _ in range(N)]
    for w, u, v in edges:
        if union(u, v):
            # add to MST
            adj[u].append((v, w))
            adj[v].append((u, w))

    # LCA preprocess
    LOG = (N+1).bit_length()
    up = [[-1]*N for _ in range(LOG)]
    minEdge = [[10**9+5]*N for _ in range(LOG)]
    depth = [0]*N

    # DFS/BFS from every root to fill parent[0], minEdge[0], depth
    visited = [False]*N
    from collections import deque
    for s in range(N):
        if not visited[s]:
            visited[s] = True
            depth[s] = 0
            up[0][s] = -1
            minEdge[0][s] = 10**9+5  # no edge to parent
            dq = deque([s])
            while dq:
                u = dq.popleft()
                for v, w in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        depth[v] = depth[u] + 1
                        up[0][v] = u
                        minEdge[0][v] = w
                        dq.append(v)

    # build binary lifting tables
    for k in range(1, LOG):
        puk = up[k-1]
        pik = up[k]
        mek = minEdge[k]
        mek1 = minEdge[k-1]
        for v in range(N):
            pv = puk[v]
            if pv < 0:
                pik[v] = -1
                mek[v] = mek1[v]
            else:
                pik[v] = puk[pv]
                mek[v] = mek1[v] if mek1[v] < minEdge[k-1][pv] else minEdge[k-1][pv]

    def lca_min_edge(u, v):
        # if not same component, return 0
        ru = find(u)
        rv = find(v)
        if ru != rv:
            return 0
        # bring to same depth
        me = 10**9+5
        if depth[u] < depth[v]:
            u, v = v, u
        diff = depth[u] - depth[v]
        for k in range(LOG):
            if diff & (1<<k):
                if minEdge[k][u] < me:
                    me = minEdge[k][u]
                u = up[k][u]
        if u == v:
            return me
        for k in range(LOG-1, -1, -1):
            if up[k][u] != up[k][v]:
                if minEdge[k][u] < me:
                    me = minEdge[k][u]
                if minEdge[k][v] < me:
                    me = minEdge[k][v]
                u = up[k][u]
                v = up[k][v]
        # last step to LCA
        if minEdge[0][u] < me:
            me = minEdge[0][u]
        if minEdge[0][v] < me:
            me = minEdge[0][v]
        return me

    # process queries
    Q = int(input())
    out = []
    for _ in range(Q):
        A, B, Y, C, D, Z = map(int, input().split())
        u = node(A-1, B-1)
        v = node(C-1, D-1)
        T = lca_min_edge(u, v)
        # compute cost
        if T >= min(Y, Z):
            cost = abs(Y - Z)
        else:
            cost = Y + Z - 2*T
        out.append(str(cost))

    print("
".join(out))

if __name__ == "__main__":
    main()