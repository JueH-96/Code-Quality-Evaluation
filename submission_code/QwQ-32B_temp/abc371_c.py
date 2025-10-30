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
    
    # Initialize A matrix
    A = [[0] * (N + 1) for _ in range(N + 1)]  # 1-based indexing
    
    for i in range(1, N):  # i from 1 to N-1 inclusive
        parts = list(map(int, sys.stdin.readline().split()))
        for idx in range(len(parts)):
            j = i + 1 + idx
            A[i][j] = parts[idx]
    
    min_cost = float('inf')
    
    for perm in itertools.permutations(range(1, N + 1)):
        target = set()
        for (u, v) in G_edges:
            a = perm[u - 1]
            b = perm[v - 1]
            if a > b:
                a, b = b, a
            target.add((a, b))
        
        cost = 0
        for i in range(1, N):
            for j in range(i + 1, N + 1):
                in_target = (i, j) in target
                in_H = (i, j) in H_edges
                if in_target != in_H:
                    cost += A[i][j]
        
        if cost < min_cost:
            min_cost = cost
    
    print(min_cost)

if __name__ == "__main__":
    main()