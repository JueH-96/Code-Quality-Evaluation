import sys
import threading
def main():
    import sys
    data = sys.stdin
    INF = 10**30
    # Read input
    line = data.readline().split()
    N = int(line[0]); M = int(line[1]); Q = int(line[2])
    A = [0]*M; B = [0]*M; C = [0]*M
    for i in range(M):
        a,b,c = data.readline().split()
        A[i] = int(a)-1; B[i] = int(b)-1; C[i] = int(c)
    # Read queries
    queries = []
    deleted = [False]*M
    for _ in range(Q):
        tmp = data.readline().split()
        t = int(tmp[0])
        if t == 1:
            ei = int(tmp[1]) - 1
            queries.append((1, ei))
            deleted[ei] = True
        else:
            x = int(tmp[1]) - 1; y = int(tmp[2]) - 1
            queries.append((2, x, y))
    # Build initial distance matrix for the final graph (after all deletions)
    dist = [[INF]*N for _ in range(N)]
    for i in range(N):
        dist[i][i] = 0
    for e in range(M):
        if not deleted[e]:
            u = A[e]; v = B[e]; w = C[e]
            if w < dist[u][v]:
                dist[u][v] = w
                dist[v][u] = w
    # Floyd-Warshall
    for k in range(N):
        dk = dist[k]
        for i in range(N):
            dik = dist[i][k]
            if dik == INF:
                continue
            rowi = dist[i]
            sumik = dik
            # unroll local for speed
            for j in range(N):
                val = sumik + dk[j]
                if val < rowi[j]:
                    rowi[j] = val
    # Process queries in reverse, handling additions and answering queries
    ans_rev = []
    for q in reversed(queries):
        if q[0] == 2:
            # shortest path query
            _, x, y = q
            d = dist[x][y]
            ans_rev.append(str(d if d < INF else -1))
        else:
            # edge addition in reverse
            _, ei = q
            u = A[ei]; v = B[ei]; w = C[ei]
            # snapshot the two rows needed for update
            old_u = dist[u][:]
            old_v = dist[v][:]
            # update all pairs
            # using locals for speed
            Nloc = N
            dloc = dist
            wloc = w
            ou = old_u
            ov = old_v
            for i in range(Nloc):
                rowi = dloc[i]
                bu = ou[i] + wloc  # old[i][u] + w
                bv = ov[i] + wloc  # old[i][v] + w
                # inner loop
                # we reference ou[j] and ov[j]
                for j in range(Nloc):
                    # two possible paths via new edge
                    # path i->u->v->j : bu + old[v][j] => bu + ov[j]
                    # path i->v->u->j : bv + old[u][j] => bv + ou[j]
                    # pick min of these two
                    c1 = bu + ov[j]
                    c2 = bv + ou[j]
                    nc = c1 if c1 < c2 else c2
                    if nc < rowi[j]:
                        rowi[j] = nc
    # Output answers in original order
    out = sys.stdout
    for v in reversed(ans_rev):
        out.write(v + "
")

if __name__ == "__main__":
    main()