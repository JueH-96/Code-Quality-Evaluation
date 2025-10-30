N, K, P = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(N)]
C = [int(input()) for _ in range(N)]

dp = [float('inf')] * (P + 1)
dp[0] = 0

for i in range(N):
    for j in range(K):
        for p in range(P, A[i][j] - 1, -1):
            dp[p] = min(dp[p], dp[p - A[i][j]] + C[i])

ans = min(dp[p] for p in range(P, -1, -1) if p >= P - dp[p])
print(ans if ans != float('inf') else -1)