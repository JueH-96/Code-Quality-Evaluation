import itertools

def read_input():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    
    M_G = int(data[idx])
    idx += 1
    
    G = [[0] * N for _ in range(N)]
    for _ in range(M_G):
        u = int(data[idx]) - 1
        v = int(data[idx + 1]) - 1
        G[u][v] = 1
        G[v][u] = 1
        idx += 2
    
    M_H = int(data[idx])
    idx += 1
    
    H = [[0] * N for _ in range(N)]
    for _ in range(M_H):
        a = int(data[idx]) - 1
        b = int(data[idx + 1]) - 1
        H[a][b] = 1
        H[b][a] = 1
        idx += 2
    
    A = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            A[i][j] = int(data[idx])
            A[j][i] = A[i][j]
            idx += 1
    
    return N, G, H, A

def calculate_cost(G, H, A, perm):
    N = len(G)
    cost = 0
    for i in range(N):
        for j in range(i + 1, N):
            if G[i][j] != H[perm[i]][perm[j]]:
                cost += A[perm[i]][perm[j]]
    return cost

def main():
    N, G, H, A = read_input()
    
    min_cost = float('inf')
    for perm in itertools.permutations(range(N)):
        cost = calculate_cost(G, H, A, perm)
        if cost < min_cost:
            min_cost = cost
    
    print(min_cost)

if __name__ == "__main__":
    main()