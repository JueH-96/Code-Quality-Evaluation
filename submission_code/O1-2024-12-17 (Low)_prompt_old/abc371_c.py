def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Parse input
    # N
    idx = 0
    N = int(input_data[idx]); idx += 1
    
    # M_G and edges for G
    M_G = int(input_data[idx]); idx += 1
    G_edges = set()
    for _ in range(M_G):
        u = int(input_data[idx]) - 1
        v = int(input_data[idx+1]) - 1
        idx += 2
        if u > v:
            u, v = v, u
        G_edges.add((u, v))
    
    # M_H and edges for H
    M_H = int(input_data[idx]); idx += 1
    H_edges = set()
    for _ in range(M_H):
        a = int(input_data[idx]) - 1
        b = int(input_data[idx+1]) - 1
        idx += 2
        if a > b:
            a, b = b, a
        H_edges.add((a, b))
    
    # Cost matrix A_{i,j} for i<j
    # We'll store it in a 2D list A, where A[i][j] is valid if i<j
    A = [[0]*N for _ in range(N)]
    # We read A_{1,2}, A_{1,3}, ..., A_{1,N}, then A_{2,3}, etc.
    # In 0-based, that's A[0][1], A[0][2],..., A[0][N-1], then A[1][2], ...
    for i in range(N-1):
        for j in range(i+1, N):
            A[i][j] = int(input_data[idx])
            idx += 1
    
    # If N=1, no edges to toggle; either G and H are trivially isomorphic.
    if N == 1:
        print(0)
        return
    
    # Precompute cost for toggling each pair (i, j) based on difference
    # We'll define a function cost_of_permutation(P) that returns
    # the sum of costs needed to ensure that "G is isomorphic to H under P".
    # Under an isomorphism P: if (i,j) in G_edges <=> (P[i],P[j]) should be in H_edges.
    # So if there's a mismatch, we pay A[min(P[i],P[j])][max(P[i],P[j])].
    
    from math import inf
    from itertools import permutations
    
    # Convert sets to quick membership checks
    # G_edges, H_edges already sets of (u, v) with u<v.
    
    def cost_of_permutation(P):
        c = 0
        for i in range(N):
            for j in range(i+1, N):
                in_g = ((i, j) in G_edges)
                # The mapped pair is (P[i], P[j]) in H
                p_i, p_j = P[i], P[j]
                if p_i > p_j:
                    p_i, p_j = p_j, p_i
                in_h = ((p_i, p_j) in H_edges)
                if in_g != in_h:  # mismatch => toggle needed
                    c += A[p_i][p_j]
        return c
    
    # Try all permutations of range(N)
    # N<=8 => 40320 permutations in worst case
    # We'll just try them all and take the minimum cost
    min_cost = inf
    
    from itertools import permutations
    for P in permutations(range(N)):
        val = cost_of_permutation(P)
        if val < min_cost:
            min_cost = val
    
    print(min_cost)


def main():
    solve()

if __name__ == "__main__":
    main()