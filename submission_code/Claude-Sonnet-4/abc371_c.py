from itertools import permutations

def solve():
    N = int(input())
    
    # Read graph G
    M_G = int(input())
    G = set()
    for _ in range(M_G):
        u, v = map(int, input().split())
        G.add((min(u, v), max(u, v)))
    
    # Read graph H
    M_H = int(input())
    H = set()
    for _ in range(M_H):
        a, b = map(int, input().split())
        H.add((min(a, b), max(a, b)))
    
    # Read cost matrix A
    A = {}
    for i in range(1, N):
        costs = list(map(int, input().split()))
        for j in range(len(costs)):
            A[(i, i + j + 1)] = costs[j]
    
    min_cost = float('inf')
    
    # Try all permutations
    for perm in permutations(range(1, N + 1)):
        cost = 0
        
        # For each possible edge (i, j) where i < j
        for i in range(1, N + 1):
            for j in range(i + 1, N + 1):
                # Check if edge (i, j) exists in G
                edge_in_G = (i, j) in G
                
                # Check if corresponding edge exists in H under this permutation
                # perm[i-1] and perm[j-1] are the vertices in H that correspond to i and j in G
                pi, pj = perm[i-1], perm[j-1]
                edge_in_H = (min(pi, pj), max(pi, pj)) in H
                
                # If they don't match, we need to add/remove edge in H
                if edge_in_G != edge_in_H:
                    cost += A[(min(pi, pj), max(pi, pj))]
        
        min_cost = min(min_cost, cost)
    
    print(min_cost)

solve()