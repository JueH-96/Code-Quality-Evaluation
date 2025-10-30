import sys
from itertools import permutations

def main():
    N = int(sys.stdin.readline())

    M_G = int(sys.stdin.readline())
    G_edges = []
    for _ in range(M_G):
        u, v = map(int, sys.stdin.readline().split())
        if u > v:
            u, v = v, u
        G_edges.append((u, v))

    M_H = int(sys.stdin.readline())
    H_edges = set()
    for _ in range(M_H):
        a, b = map(int, sys.stdin.readline().split())
        if a > b:
            a, b = b, a
        H_edges.add((a, b))

    # Read cost matrix A
    A = [[0] * (N + 2) for _ in range(N + 2)]  # Using 1-based indexing
    for i in range(1, N):  # i from 1 to N-1
        line = list(map(int, sys.stdin.readline().split()))
        current_j = i + 1
        for val in line:
            A[i][current_j] = val
            current_j += 1

    min_cost = float('inf')

    # Generate all permutations of the nodes 1..N
    for perm in permutations(range(1, N + 1)):
        S = set()
        for (u, v) in G_edges:
            x = perm[u - 1]
            y = perm[v - 1]
            if x > y:
                x, y = y, x
            S.add((x, y))
        
        sym_diff = S.symmetric_difference(H_edges)
        cost = 0
        for (x, y) in sym_diff:
            cost += A[x][y]
        
        if cost < min_cost:
            min_cost = cost

    print(min_cost)

if __name__ == "__main__":
    main()