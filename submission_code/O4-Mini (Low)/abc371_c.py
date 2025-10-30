def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    # Read graph G
    M_G = int(input())
    G = [[0]*N for _ in range(N)]
    for _ in range(M_G):
        u, v = map(int, input().split())
        u -= 1; v -= 1
        G[u][v] = 1
        G[v][u] = 1

    # Read graph H
    M_H = int(input())
    H = [[0]*N for _ in range(N)]
    for _ in range(M_H):
        a, b = map(int, input().split())
        a -= 1; b -= 1
        H[a][b] = 1
        H[b][a] = 1

    # Read cost matrix A (upper triangle)
    A = [[0]*N for _ in range(N)]
    for i in range(N-1):
        row = list(map(int, input().split()))
        # row has length N - i - 1, corresponding to A_{i, j} for j = i+1 .. N-1
        for k, c in enumerate(row, start=i+1):
            A[i][k] = c
            A[k][i] = c

    # Try all permutations P of [0..N-1] mapping G's vertex i to H's vertex P[i].
    from itertools import permutations
    INF = 10**30
    ans = INF

    # For each permutation, compute cost to flip edges in H so that
    # for all i<j: G[i][j] == H[P[i]][P[j]]. If they differ, flip edge (P[i],P[j]) at cost A[P[i]][P[j]].
    for P in permutations(range(N)):
        total = 0
        # early prune
        for i in range(N):
            pi = P[i]
            # optional small optimization: compare degree? but N<=8 so skip
            pass
        # compute full cost
        c = 0
        for i in range(N):
            pi = P[i]
            for j in range(i+1, N):
                pj = P[j]
                if G[i][j] != H[pi][pj]:
                    c += A[pi][pj]
                    # small prune
                    if c >= ans:
                        break
            if c >= ans:
                break
        else:
            # only if inner loops didn't break
            if c < ans:
                ans = c

    print(ans)

if __name__ == "__main__":
    main()