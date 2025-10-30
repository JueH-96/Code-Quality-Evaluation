import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    # Read G
    M_G = int(next(it))
    G_adj = [[False]*N for _ in range(N)]
    for _ in range(M_G):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        G_adj[u][v] = True
        G_adj[v][u] = True
    # Read H
    M_H = int(next(it))
    H_adj = [[False]*N for _ in range(N)]
    for _ in range(M_H):
        a = int(next(it)) - 1
        b = int(next(it)) - 1
        H_adj[a][b] = True
        H_adj[b][a] = True
    # Read costs A_{i,j} for 1<=i<j<=N
    A = [[0]*N for _ in range(N)]
    for i in range(N-1):
        # next line has N-i-1 entries
        for j in range(i+1, N):
            c = int(next(it))
            A[i][j] = c
            A[j][i] = c

    # Prebuild list of all unordered pairs (i,j) i<j
    pairs = [(i,j) for i in range(N) for j in range(i+1, N)]

    # We'll try all permutations P of [0..N-1], where P[i] = index in H to map G-vertex i
    from itertools import permutations
    INF = 10**30
    best = INF

    # For speed, bind locals
    Gm = G_adj
    Hm = H_adj
    Ac = A
    pr = pairs
    Np = N

    for P in permutations(range(Np)):
        cost = 0
        # Early prune
        # iterate over all pairs
        for (i,j) in pr:
            gi = Gm[i][j]
            hij = Hm[P[i]][P[j]]
            if gi != hij:
                cost += Ac[P[i]][P[j]]
                if cost >= best:
                    break
        else:
            # only executed if inner loop wasn't broken
            if cost < best:
                best = cost

    # If best remains INF, no mapping found, but that can't happen
    if best == INF:
        best = 0
    sys.stdout.write(str(best))

if __name__ == "__main__":
    main()