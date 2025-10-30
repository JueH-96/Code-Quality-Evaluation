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
    
    # Precompute the cost matrix
    cost = [[0] * (N+1) for _ in range(N+1)]
    idx = 0
    for i in range(1, N):
        for j in range(i+1, N+1):
            cost[i][j] = A[i-1][j-i-1]
            cost[j][i] = cost[i][j]
    
    min_cost = float('inf')
    # Iterate over all possible permutations
    for perm in itertools.permutations(range(1, N+1)):
        total_cost = 0
        # Create the mapping
        mapping = {i+1: perm[i] for i in range(N)}
        # Iterate over all possible edges in G
        for u in range(1, N+1):
            for v in range(u+1, N+1):
                # Check if (u,v) is in G
                if (u, v) in G_edges:
                    # Check if (mapping[u], mapping[v]) is in H
                    if (mapping[u], mapping[v]) not in H_edges and (mapping[v], mapping[u]) not in H_edges:
                        total_cost += cost[mapping[u]][mapping[v]]
                else:
                    # Check if (mapping[u], mapping[v]) is in H
                    if (mapping[u], mapping[v]) in H_edges or (mapping[v], mapping[u]) in H_edges:
                        total_cost += cost[mapping[u]][mapping[v]]
        if total_cost < min_cost:
            min_cost = total_cost
        if min_cost == 0:
            break
    print(min_cost)

if __name__ == "__main__":
    main()