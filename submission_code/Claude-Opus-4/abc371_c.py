from itertools import permutations

# Read input
N = int(input())

# Read graph G
M_G = int(input())
G = [[False] * (N+1) for _ in range(N+1)]
for _ in range(M_G):
    u, v = map(int, input().split())
    G[u][v] = True
    G[v][u] = True

# Read graph H
M_H = int(input())
H = [[False] * (N+1) for _ in range(N+1)]
for _ in range(M_H):
    a, b = map(int, input().split())
    H[a][b] = True
    H[b][a] = True

# Read costs
A = [[0] * (N+1) for _ in range(N+1)]
for i in range(1, N):
    costs = list(map(int, input().split()))
    for j in range(len(costs)):
        A[i][i+j+1] = costs[j]
        A[i+j+1][i] = costs[j]

# Try all permutations
min_cost = float('inf')

for perm in permutations(range(1, N+1)):
    # perm[i-1] is where vertex i maps to
    cost = 0
    
    # For each edge position in G
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            # Check if edge (i,j) exists in G
            g_has_edge = G[i][j]
            
            # Check if corresponding edge exists in H after permutation
            # Vertex i maps to perm[i-1], vertex j maps to perm[j-1]
            pi = perm[i-1]
            pj = perm[j-1]
            h_has_edge = H[pi][pj]
            
            # If they differ, we need to add/remove edge in H
            if g_has_edge != h_has_edge:
                # Cost is A[pi][pj] but we need to ensure pi < pj for indexing
                if pi > pj:
                    pi, pj = pj, pi
                cost += A[pi][pj]
    
    min_cost = min(min_cost, cost)

print(min_cost)