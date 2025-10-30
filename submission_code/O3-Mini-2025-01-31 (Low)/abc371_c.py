def main():
    import sys, itertools
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    
    # Read graph G.
    MG = int(next(it))
    # Build an (N+1)x(N+1) adjacency matrix for G.
    G = [[0] * (N + 1) for _ in range(N + 1)]
    for _ in range(MG):
        u = int(next(it))
        v = int(next(it))
        G[u][v] = 1
        G[v][u] = 1
    
    # Read graph H.
    MH = int(next(it))
    H = [[0] * (N + 1) for _ in range(N + 1)]
    for _ in range(MH):
        u = int(next(it))
        v = int(next(it))
        H[u][v] = 1
        H[v][u] = 1
        
    # Read cost matrix A for pairs (i,j) with 1 <= i < j <= N.
    # We will store it in a matrix A (symmetric, though we only need i<j).
    A = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N):
        for j in range(i + 1, N + 1):
            cost = int(next(it))
            A[i][j] = cost
            A[j][i] = cost  # store symmetrically for convenience

    # Explanation:
    # We are allowed to toggle any edge (i,j) in H with cost A[i][j] if needed.
    # Two graphs G and H are isomorphic if there exists a permutation P (of vertices from 1 to N)
    # such that for all 1<=i<j<=N, we have:
    #     G has an edge between i and j if and only if H has an edge between P[i] and P[j].
    # We use a candidate permutation mapping from vertex i in G to vertex perm[i-1] in H.
    # For each pair (i,j) with 1 <= i < j <= N, the desired state of the edge in H
    # between vertices perm[i-1] and perm[j-1] should equal the state in G between i and j.
    # If the current configuration in H is different, we pay cost A[min(perm[i-1], perm[j-1])][max(perm[i-1], perm[j-1])].
    
    best = float('inf')
    vertices = list(range(1, N + 1))
    # Try all permutations for the mapping.
    for perm in itertools.permutations(vertices):
        total_cost = 0
        # For each edge pair in G, verify that H (at the mapped positions) matches.
        for i in range(1, N):
            hi = perm[i - 1]
            for j in range(i + 1, N + 1):
                hj = perm[j - 1]
                desired = G[i][j]  # desired state (exists or not) as in graph G.
                current = H[hi][hj]  # current state in H between vertices hi and hj.
                if desired != current:
                    # If different, we need to toggle this edge in H.
                    # The operation cost for edge (u,v) with u < v is A[u][v].
                    u, v = hi, hj
                    if u > v:
                        u, v = v, u
                    total_cost += A[u][v]
                    # Early break if cost already exceeds the best found.
                    if total_cost >= best:
                        break
            if total_cost >= best:
                break
        # Update best answer.
        if total_cost < best:
            best = total_cost

    sys.stdout.write(str(best))


if __name__ == '__main__':
    main()