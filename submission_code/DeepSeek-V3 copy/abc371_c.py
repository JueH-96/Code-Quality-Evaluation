import sys
import itertools

def main():
    N = int(sys.stdin.readline())
    M_G = int(sys.stdin.readline())
    G_edges = set()
    for _ in range(M_G):
        u, v = map(int, sys.stdin.readline().split())
        if u > v:
            u, v = v, u
        G_edges.add((u, v))
    M_H = int(sys.stdin.readline())
    H_edges = set()
    for _ in range(M_H):
        a, b = map(int, sys.stdin.readline().split())
        if a > b:
            a, b = b, a
        H_edges.add((a, b))
    A = []
    for i in range(1, N):
        row = list(map(int, sys.stdin.readline().split()))
        A.append(row)
    
    # Create a cost matrix
    cost = [[0] * (N+1) for _ in range(N+1)]
    idx = 0
    for i in range(1, N):
        for j in range(i+1, N+1):
            cost[i][j] = A[i-1][j-i-1]
            cost[j][i] = cost[i][j]
    
    min_cost = float('inf')
    # Generate all possible permutations
    for perm in itertools.permutations(range(1, N+1)):
        total_cost = 0
        # For each edge in G, check if the corresponding edge exists in H
        for u, v in G_edges:
            pu, pv = perm[u-1], perm[v-1]
            if pu > pv:
                pu, pv = pv, pu
            if (pu, pv) not in H_edges:
                total_cost += cost[pu][pv]
        # For each edge in H, check if the corresponding edge exists in G
        for a, b in H_edges:
            pa, pb = perm[a-1], perm[b-1]
            if pa > pb:
                pa, pb = pb, pa
            if (pa, pb) not in G_edges:
                total_cost += cost[pa][pb]
        if total_cost < min_cost:
            min_cost = total_cost
    print(min_cost)

if __name__ == "__main__":
    main()