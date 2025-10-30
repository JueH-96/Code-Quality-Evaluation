import sys
import threading
def main():
    import sys
    input = sys.stdin.readline
    N, M, Q = map(int, input().split())
    A = [0]*M
    B = [0]*M
    C = [0]*M
    for i in range(M):
        a,b,c = map(int, input().split())
        A[i]=a-1; B[i]=b-1; C[i]=c
    queries = []
    closed = [False]*M
    for _ in range(Q):
        t,*rest = map(int, input().split())
        if t==1:
            i = rest[0]-1
            queries.append((1,i))
            closed[i]=True
        else:
            x,y = rest
            queries.append((2,x-1,y-1))
    # build final graph: only edges never closed or already re-added by reverse adds
    INF = 10**30
    dist = [[INF]*N for _ in range(N)]
    for i in range(N):
        dist[i][i]=0
    for i in range(M):
        if not closed[i]:
            u = A[i]; v = B[i]; c = C[i]
            if c < dist[u][v]:
                dist[u][v] = c
                dist[v][u] = c
    # initial APSP by Floyd-Warshall
    for k in range(N):
        dk = dist[k]
        for i in range(N):
            di = dist[i]
            ik = di[k]
            if ik==INF: continue
            # di[j] = min(di[j], ik + dk[j])
            for j in range(N):
                val = ik + dk[j]
                if val < di[j]:
                    di[j] = val
    # process queries in reverse
    answers = []
    for q in reversed(queries):
        if q[0]==2:
            x,y = q[1], q[2]
            d = dist[x][y]
            answers.append(d if d<INF else -1)
        else:
            # reverse of closure: add edge back
            idx = q[1]
            u = A[idx]; v = B[idx]; c = C[idx]
            # if this edge already better, skip updates?
            # but still must propagate
            # do the standard addition update
            # for all i,j: dist[i][j] = min(dist[i][j], dist[i][u]+c+dist[v][j], dist[i][v]+c+dist[u][j])
            du = dist[u]
            dv = dist[v]
            for i in range(N):
                di = dist[i]
                iu = di[u]
                iv = di[v]
                # if both INF skip
                if iu==INF and iv==INF:
                    continue
                # For each j:
                # try path i->u->v->j
                # and i->v->u->j
                # we can do two arrays du and dv
                base1 = iu + c
                base2 = iv + c
                # now for j
                # dr1_j = base1 + dv[j]
                # dr2_j = base2 + du[j]
                # new = min(di[j], dr1_j, dr2_j)
                for j in range(N):
                    v1 = base1 + dv[j]
                    if v1 < di[j]:
                        di[j] = v1
                    v2 = base2 + du[j]
                    if v2 < di[j]:
                        di[j] = v2
    # reverse answers to original order
    answers.reverse()
    out = sys.stdout
    idx = 0
    for q in queries:
        if q[0]==2:
            out.write(str(answers[idx]) + "
")
            idx += 1

if __name__ == "__main__":
    main()