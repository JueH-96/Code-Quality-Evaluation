from itertools import permutations

def solve():
    N = int(input())
    
    # Read graph G
    M_G = int(input())
    G = [[False] * N for _ in range(N)]
    for _ in range(M_G):
        u, v = map(int, input().split())
        u -= 1  # Convert to 0-indexed
        v -= 1
        G[u][v] = True
        G[v][u] = True
    
    # Read graph H
    M_H = int(input())
    H = [[False] * N for _ in range(N)]
    for _ in range(M_H):
        a, b = map(int, input().split())
        a -= 1  # Convert to 0-indexed
        b -= 1
        H[a][b] = True
        H[b][a] = True
    
    # Read cost matrix A
    A = [[0] * N for _ in range(N)]
    for i in range(N - 1):
        costs = list(map(int, input().split()))
        for j in range(len(costs)):
            A[i][i + j + 1] = costs[j]
            A[i + j + 1][i] = costs[j]  # Make it symmetric
    
    min_cost = float('inf')
    
    # Try all permutations
    for perm in permutations(range(N)):
        cost = 0
        for i in range(N):
            for j in range(i + 1, N):
                # Check if edge (i, j) exists in G
                edge_in_G = G[i][j]
                # Check if edge (perm[i], perm[j]) exists in H
                pi, pj = perm[i], perm[j]
                edge_in_H = H[pi][pj]
                
                if edge_in_G != edge_in_H:
                    # Need to add or remove edge (pi, pj) in H
                    cost += A[min(pi, pj)][max(pi, pj)]
        
        min_cost = min(min_cost, cost)
    
    print(min_cost)

solve()