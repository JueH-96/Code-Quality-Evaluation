import sys
import itertools

def main():
    lines = sys.stdin.read().splitlines()
    idx = 0
    N = int(lines[idx].strip())
    idx += 1
    if N == 1:
        print(0)
        return
    M_G = int(lines[idx].strip())
    idx += 1
    adj_G = [[0] * N for _ in range(N)]
    for _ in range(M_G):
        u, v = map(int, lines[idx].strip().split())
        adj_G[u-1][v-1] = 1
        adj_G[v-1][u-1] = 1
        idx += 1
    M_H = int(lines[idx].strip())
    idx += 1
    adj_H = [[0] * N for _ in range(N)]
    for _ in range(M_H):
        a, b = map(int, lines[idx].strip().split())
        adj_H[a-1][b-1] = 1
        adj_H[b-1][a-1] = 1
        idx += 1
    A = [[0] * N for _ in range(N)]
    for k in range(N-1):
        vals = list(map(int, lines[idx].strip().split()))
        for m in range(len(vals)):
            j = k + m + 1
            A[k][j] = vals[m]
            A[j][k] = vals[m]
        idx += 1
    min_cost = float('inf')
    vertices = list(range(N))
    for P in itertools.permutations(vertices):
        adj_HP = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                adj_HP[i][j] = adj_H[P[i]][P[j]]
        cost = 0
        for i in range(N):
            for j in range(i+1, N):
                if adj_G[i][j] != adj_HP[i][j]:
                    cost += A[i][j]
        if cost < min_cost:
            min_cost = cost
    print(min_cost)

if __name__ == '__main__':
    main()