def main():
    import sys, itertools
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    
    # Build graph G as an N x N boolean matrix.
    G = [[False] * N for _ in range(N)]
    M_G = int(next(it))
    for _ in range(M_G):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        G[u][v] = True
        G[v][u] = True

    # Build graph H as an N x N boolean matrix.
    H = [[False] * N for _ in range(N)]
    M_H = int(next(it))
    for _ in range(M_H):
        a = int(next(it)) - 1
        b = int(next(it)) - 1
        H[a][b] = True
        H[b][a] = True

    # Build cost matrix A. It is given for pairs (i, j) with 1 <= i < j <= N.
    # We store it in a symmetric N x N matrix.
    A = [[0] * N for _ in range(N)]
    for i in range(N - 1):
        for j in range(i + 1, N):
            cost = int(next(it))
            A[i][j] = cost
            A[j][i] = cost

    # We want to update H (by toggling edges at cost A[i][j] per toggle)
    # such that the resulting H' is isomorphic to G.
    # Graphs G and H are isomorphic if there exists a permutation P of vertices
    # such that for every pair (i, j) with i < j, an edge exists between vertices i and j in G
    # if and only if an edge exists between vertices P[i] and P[j] in H.
    #
    # For each permutation P (of indices 0...N-1) we view it as mapping:
    # vertex i (of G) -> vertex P[i] (of H).
    # Then for every pair (i, j), i < j, the edge status in H for pair (P[i], P[j])
    # must equal the edge status of G at (i, j). Since our allowed operations are independent
    # on each pair, the cost to "fix" an edge (u, v) is A[min(u,v)][max(u,v)] if the current
    # status in H differs from the required one.
    
    best_cost = 10**18
    for perm in itertools.permutations(range(N)):
        cost = 0
        valid = True
        # For every pair (i, j) (with i<j) in graph G, check the corresponding pair in H.
        for i in range(N):
            u = perm[i]
            for j in range(i + 1, N):
                v = perm[j]
                desired = G[i][j]
                current = H[u][v]
                if desired != current:
                    # We need to toggle this edge in H.
                    # Determine cost using the given A values.
                    if u < v:
                        cost += A[u][v]
                    else:
                        cost += A[v][u]
                    if cost >= best_cost:
                        valid = False
                        break
            if not valid:
                break
        if valid and cost < best_cost:
            best_cost = cost

    sys.stdout.write(str(best_cost))

if __name__ == '__main__':
    main()