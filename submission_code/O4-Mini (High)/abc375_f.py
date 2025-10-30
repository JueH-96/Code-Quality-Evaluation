import sys
import threading
def main():
    import sys
    input = sys.stdin.readline
    N, M, Q = map(int, input().split())
    A = [0] * M
    B = [0] * M
    C = [0] * M
    for i in range(M):
        a, b, c = map(int, input().split())
        A[i] = a - 1
        B[i] = b - 1
        C[i] = c
    will_close = [False] * M
    queries = [None] * Q
    for qi in range(Q):
        data = input().split()
        if data[0] == '1':
            idx = int(data[1]) - 1
            will_close[idx] = True
            queries[qi] = (1, idx)
        else:
            x = int(data[1]) - 1
            y = int(data[2]) - 1
            queries[qi] = (2, x, y)
    inf = 10**30
    # Initialize distance matrix
    dist = [[inf] * N for _ in range(N)]
    for i in range(N):
        dist[i][i] = 0
    # Add all edges that never get closed
    for i in range(M):
        if not will_close[i]:
            u = A[i]; v = B[i]; w = C[i]
            if w < dist[u][v]:
                dist[u][v] = w
                dist[v][u] = w
    # Floyd-Warshall for the initial graph
    for k in range(N):
        dk = dist[k]
        for i in range(N):
            di = dist[i]
            ik = di[k]
            if ik == inf:
                continue
            base = ik
            # try to improve dist[i][j] via k
            for j in range(N):
                val = base + dk[j]
                if val < di[j]:
                    di[j] = val
    # Process queries in reverse, handling edge-adds and distance queries
    ans = []
    # Local aliases for speed
    dist_mat = dist
    N_local = N
    inf_local = inf
    for q in reversed(queries):
        if q[0] == 2:
            _, x, y = q
            d = dist_mat[x][y]
            ans.append(str(d if d < inf_local else -1))
        else:
            _, eidx = q
            u = A[eidx]; v = B[eidx]; w = C[eidx]
            # If this edge cannot improve the u-v distance, skip
            if w >= dist_mat[u][v]:
                continue
            du = dist_mat[u]
            dv = dist_mat[v]
            # Dynamic APSP update for this new edge
            for i in range(N_local):
                di = dist_mat[i]
                diu = di[u]; div = di[v]
                if diu == inf_local and div == inf_local:
                    continue
                bu = diu
                bv = div
                # try all j
                for j in range(N_local):
                    # path i->u->v->j or i->v->u->j
                    c1 = bu + dv[j]
                    c2 = bv + du[j]
                    c = c1 if c1 <= c2 else c2
                    if c < di[j]:
                        di[j] = c
    # Reverse collected answers and print
    ans.reverse()
    sys.stdout.write("
".join(ans))

if __name__ == "__main__":
    main()