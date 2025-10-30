import itertools

def solve():
    N = int(input())
    G = [[0]*N for _ in range(N)]
    H = [[0]*N for _ in range(N)]
    A = [[0]*N for _ in range(N)]
    MG = int(input())
    for _ in range(MG):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        G[u][v] = G[v][u] = 1
    MH = int(input())
    for _ in range(MH):
        a, b = map(int, input().split())
        a -= 1
        b -= 1
        H[a][b] = H[b][a] = 1
    for i in range(N-1):
        A[i][i+1:] = list(map(int, input().split()))
        for j in range(i+1, N):
            A[j][i] = A[i][j]
    ans = float('inf')
    for P in itertools.permutations(range(N)):
        cost = 0
        for i in range(N):
            for j in range(i+1, N):
                if G[i][j] != H[P[i]][P[j]]:
                    cost += A[P[i]][P[j]]
        ans = min(ans, cost)
    print(ans)

solve()