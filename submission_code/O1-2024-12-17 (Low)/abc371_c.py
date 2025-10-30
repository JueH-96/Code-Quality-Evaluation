def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    idx = 0

    # Read N
    N = int(input_data[idx]); idx+=1
    # Read M_G
    M_G = int(input_data[idx]); idx+=1

    # Build adjacency matrix for G
    G = [[False]*N for _ in range(N)]
    for _ in range(M_G):
        u = int(input_data[idx]); idx+=1
        v = int(input_data[idx]); idx+=1
        # Convert to zero-based
        u -= 1
        v -= 1
        G[u][v] = True
        G[v][u] = True

    # Read M_H
    M_H = int(input_data[idx]); idx+=1

    # Build adjacency matrix for H
    H = [[False]*N for _ in range(N)]
    for _ in range(M_H):
        a = int(input_data[idx]); idx+=1
        b = int(input_data[idx]); idx+=1
        # Convert to zero-based
        a -= 1
        b -= 1
        H[a][b] = True
        H[b][a] = True

    # Read cost matrix A in triangular form
    # A[i][j] = cost of toggling edge between i,j (i < j)
    A = [[0]*N for _ in range(N)]
    for i in range(N-1):
        for j in range(i+1, N):
            cost_val = int(input_data[idx]); idx+=1
            A[i][j] = cost_val
            A[j][i] = cost_val

    # We'll check all permutations of {0,1,...,N-1}
    from itertools import permutations

    # Precompute G[i][j] for i<j in a list so we only check i<j pairs
    # but simpler is to check all i<j directly in loops.
    min_cost = float('inf')

    for perm in permutations(range(N)):
        cost_perm = 0
        # Compare adjacency under perm
        for i in range(N):
            for j in range(i+1, N):
                if G[i][j] != H[perm[i]][perm[j]]:
                    # We need to toggle the edge in H to match G
                    cost_perm += A[perm[i]][perm[j]]
                    # If cost already exceeds the best, we can break early
                    if cost_perm >= min_cost:
                        break
            if cost_perm >= min_cost:
                break
        if cost_perm < min_cost:
            min_cost = cost_perm

    print(min_cost)

# Do not forget to call main()
if __name__ == "__main__":
    main()