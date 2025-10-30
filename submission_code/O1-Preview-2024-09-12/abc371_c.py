# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import math
    import itertools
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    M_G = int(sys.stdin.readline())
    # Build adjacency matrix for G
    G = [[0]*N for _ in range(N)]
    for _ in range(M_G):
        u,v = map(int, sys.stdin.readline().split())
        u -=1 ; v -=1
        G[u][v] = 1
        G[v][u] = 1

    M_H = int(sys.stdin.readline())
    # Build adjacency matrix for H
    H = [[0]*N for _ in range(N)]
    for _ in range(M_H):
        a,b = map(int, sys.stdin.readline().split())
        a -=1 ; b -=1
        H[a][b] = 1
        H[b][a] = 1

    # Read A_{i,j}, stored as A[i][j] for i<j
    A = [[0]*N for _ in range(N)]
    for i in range(N-1):
        line = sys.stdin.readline().split()
        for idx, j in enumerate(range(i+1,N)):
            A[i][j] = int(line[idx])

    min_total_cost = float('inf')

    # Generate all permutations of [0..N-1]
    for P in itertools.permutations(range(N)):
        total_cost = 0
        over = False
        for i in range(N):
            if over:
                break
            for j in range(i+1,N):
                edge_in_G = G[i][j]
                # Nodes in H are P[0..N-1]
                u = P[i]
                v = P[j]
                edge_in_H = H[u][v]

                if edge_in_G != edge_in_H:
                    # Need to toggle edge between u and v in H
                    cost = A[min(u,v)][max(u,v)]
                    total_cost += cost
                    if total_cost >= min_total_cost:
                        # No need to proceed further for this permutation
                        over = True
                        break
        if total_cost < min_total_cost:
            min_total_cost = total_cost
    print(min_total_cost)



threading.Thread(target=main).start()