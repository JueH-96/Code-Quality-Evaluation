import itertools

N = int(input())
M_G = int(input())

# Build adjacency matrix for G
G = [[False] * N for _ in range(N)]
for _ in range(M_G):
    u, v = map(int, input().split())
    u -= 1  # Convert to 0-indexed
    v -= 1
    G[u][v] = True
    G[v][u] = True

M_H = int(input())

# Build adjacency matrix for H
H = [[False] * N for _ in range(N)]
for _ in range(M_H):
    a, b = map(int, input().split())
    a -= 1  # Convert to 0-indexed
    b -= 1
    H[a][b] = True
    H[b][a] = True

# Read cost matrix (upper triangular)
A = [[0] * N for _ in range(N)]
for i in range(N - 1):
    row = list(map(int, input().split()))
    for j in range(i + 1, N):
        A[i][j] = row[j - i - 1]

# Try all permutations
min_cost = float('inf')
for perm in itertools.permutations(range(N)):
    # Calculate cost for this permutation
    cost = 0
    for i in range(N):
        for j in range(i + 1, N):
            # Edge (i,j) in G should correspond to edge (perm[i], perm[j]) in H
            g_has_edge = G[i][j]
            pi = perm[i]
            pj = perm[j]
            h_has_edge = H[pi][pj]
            
            if g_has_edge != h_has_edge:
                # Need to toggle edge (pi, pj) in H
                if pi > pj:
                    pi, pj = pj, pi
                cost += A[pi][pj]
    
    min_cost = min(min_cost, cost)

print(min_cost)