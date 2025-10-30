# YOUR CODE HERE
import sys
import itertools

def main():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    
    # Read Graph G
    M_G = int(data[index])
    index += 1
    G = [[0] * N for _ in range(N)]
    for _ in range(M_G):
        u = int(data[index]) - 1
        v = int(data[index + 1]) - 1
        index += 2
        G[u][v] = 1
        G[v][u] = 1
    
    # Read Graph H
    M_H = int(data[index])
    index += 1
    H = [[0] * N for _ in range(N)]
    for _ in range(M_H):
        a = int(data[index]) - 1
        b = int(data[index + 1]) - 1
        index += 2
        H[a][b] = 1
        H[b][a] = 1
    
    # Read cost matrix A
    A = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            A[i][j] = int(data[index])
            index += 1
    
    # Try all permutations of vertices
    min_cost = float('inf')
    for perm in itertools.permutations(range(N)):
        current_cost = 0
        for i in range(N):
            for j in range(i + 1, N):
                gi = G[i][j]
                hj = H[perm[i]][perm[j]]
                if gi != hj:
                    current_cost += A[min(i, j)][max(i, j)]
        
        min_cost = min(min_cost, current_cost)
    
    print(min_cost)

if __name__ == "__main__":
    main()