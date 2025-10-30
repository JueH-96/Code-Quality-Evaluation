import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    N, Q = map(int, input().split())
    queries = [tuple(map(int, input().split())) for _ in range(Q)]
    # build segment tree size
    size = 1
    while size < N:
        size <<= 1
    # total segment tree nodes are indexed 1 .. 2*size-1
    ST_N = 2 * size
    total_nodes = ST_N + Q  # we'll use 1..ST_N-1 for seg nodes, ST_N..ST_N+Q-1 for B's
    edges = []
    # seg tree edges: from internal node k to children 2k,2k+1 with cost 0
    for k in range(1, size):
        lc = 2*k
        rc = lc + 1
        edges.append((0, k, lc))
        edges.append((0, k, rc))
    # queries edges: B_i node id = ST_N - 1 + i  (i from 1 to Q)
    # actually ST_N = 2*size, so seg nodes go 1..2*size-1
    # Let B_i = (2*size - 1) + i
    baseB = ST_N - 1
    # function to decompose [l,r] into seg nodes
    def add_range_edges(l, r, qi, cost):
        # l,r are 1-based positions in leaves [1..N]
        l = l - 1 + size
        r = r - 1 + size
        bi = baseB + qi
        while l <= r:
            if (l & 1) == 1:
                edges.append((cost, bi, l))
                l += 1
            if (r & 1) == 0:
                edges.append((cost, bi, r))
                r -= 1
            l >>= 1; r >>= 1
    for i, (L, R, C) in enumerate(queries, start=1):
        add_range_edges(L, R, i, C)
    # Kruskal
    parent = list(range(total_nodes+1))
    rank = [0]*(total_nodes+1)
    def find(x):
        while parent[x]!=x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra = find(a); rb = find(b)
        if ra==rb: return False
        if rank[ra]<rank[rb]:
            parent[ra]=rb
        else:
            parent[rb]=ra
            if rank[ra]==rank[rb]:
                rank[ra]+=1
        return True
    edges.sort(key=lambda x: x[0])
    mst_cost = 0
    cnt = 0
    for cost,u,v in edges:
        if union(u,v):
            mst_cost += cost
            cnt += 1
            # early stop if we've added enough edges
            if cnt == total_nodes-1:
                break
    # Check connectivity: all leaves (size..size+N-1) and all B nodes (baseB+1..baseB+Q) in one comp
    # pick representative of leaf 1
    if N > 0:
        rep = find(size)
    else:
        rep = None
    ok = True
    # leaves
    for j in range(N):
        if find(size + j) != rep:
            ok = False
            break
    # B nodes
    if ok:
        for i in range(1, Q+1):
            if find(baseB + i) != rep:
                ok = False
                break
    if not ok:
        print(-1)
    else:
        print(mst_cost)

if __name__ == "__main__":
    main()