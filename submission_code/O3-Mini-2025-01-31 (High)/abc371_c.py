def main():
    import sys
    from itertools import permutations

    data = sys.stdin.read().split()
    it = iter(data)
    if not data:
        return

    # Read number of vertices
    N = int(next(it))
    
    # Build the adjacency matrix for graph G (0-indexed)
    G = [[0] * N for _ in range(N)]
    M_G = int(next(it))
    for _ in range(M_G):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        G[u][v] = 1
        G[v][u] = 1

    # Build the adjacency matrix for graph H (0-indexed)
    H = [[0] * N for _ in range(N)]
    M_H = int(next(it))
    for _ in range(M_H):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        H[u][v] = 1
        H[v][u] = 1

    # Read the cost matrix A.
    # The input gives the upper‐triangular part:
    # Line 1: A[0,1], A[0,2], ..., A[0, N-1]
    # Line 2: A[1,2], A[1,3], ..., A[1, N-1]
    # etc.
    cost_matrix = [[0] * N for _ in range(N)]
    for i in range(N - 1):
        for j in range(i + 1, N):
            c = int(next(it))
            cost_matrix[i][j] = c
            cost_matrix[j][i] = c

    # Our goal is to choose a series of toggles on H (each toggle on an unordered pair (i,j)
    # costs A[i,j] and flips the bit) so that H becomes isomorphic to G.
    # Two graphs are isomorphic if there is a permutation P (of vertices 0..N-1) such that:
    #   For all 0 <= i < j < N,  G[i][j] == H'( P[i], P[j] )
    # Since toggling an edge is a binary operation (flip or no flip),
    # if for a given pair (i,j) (with i < j in G) the corresponding edge in H (at vertices 
    # (min(P[i],P[j]), max(P[i], P[j])) ) does not match G[i][j], we must pay the cost A
    # for that edge.
    #
    # Thus for each permutation P we compute:
    #   cost(P) = sum_{0 <= i < j < N} [ cost_matrix[min(P[i],P[j])][max(P[i],P[j])]  if H[min(P[i],P[j])][max(P[i],P[j])] != G[i][j] ].
    # And we want the minimum such cost.

    # Precompute pairs for vertices in G (i, j) with i < j.
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    best = 10**18
    for P in permutations(range(N)):
        cur_cost = 0
        # For every edge (i, j) in G, check the corresponding edge in H at (P[i], P[j])
        for i, j in pairs:
            target = G[i][j]  # desired edge state in the final H (must match G)
            # Find the corresponding unordered pair in H
            u, v = P[i], P[j]
            if u > v:
                u, v = v, u
            # If the current state in H doesn't match the target, a toggle is needed.
            if H[u][v] != target:
                cur_cost += cost_matrix[u][v]
                if cur_cost >= best:  # early break if cost is already too high
                    break
        if cur_cost < best:
            best = cur_cost

    sys.stdout.write(str(best))

if __name__ == '__main__':
    main()