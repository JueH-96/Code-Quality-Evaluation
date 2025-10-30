import itertools

def main():
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
        idx += 2
        G[u][v] = 1
        G[v][u] = 1
    M_H = int(data[idx])
    idx += 1
    H = [[0] * N for _ in range(N)]
    for _ in range(M_H):
        a = int(data[idx]) - 1
        b = int(data[idx + 1]) - 1
        idx += 2
        H[a][b] = 1
        H[b][a] = 1
    A = [[0] * N for _ in range(N)]
    for i in range(N - 1):
        for j in range(i + 1, N):
            A[i][j] = int(data[idx])
            idx += 1
    
    min_cost = float('inf')
    for P in itertools.permutations(range(N)):
        permuted_H = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                permuted_H[i][j] = H[P[i]][P[j]]
        cost = 0
        for i in range(N):
            for j in range(i + 1, N):
                if G[i][j] != permuted_H[i][j]:
                    cost += A[i][j]
        if cost < min_cost:
            min_cost = cost
    print(min_cost)

if __name__ == '__main__':
    main()