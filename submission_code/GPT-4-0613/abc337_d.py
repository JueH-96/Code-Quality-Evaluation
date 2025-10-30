H, W, K = map(int, input().split())
S = [list(input()) for _ in range(H)]
INF = 10**9
dp = [[INF]*W for _ in range(H)]
for i in range(H):
    for j in range(W):
        if S[i][j] == 'o':
            dp[i][j] = 0
        elif S[i][j] == '.':
            if i > 0:
                dp[i][j] = min(dp[i][j], dp[i-1][j] + 1)
            if j > 0:
                dp[i][j] = min(dp[i][j], dp[i][j-1] + 1)
for i in range(H-1, -1, -1):
    for j in range(W-1, -1, -1):
        if S[i][j] == '.':
            if i < H-1:
                dp[i][j] = min(dp[i][j], dp[i+1][j] + 1)
            if j < W-1:
                dp[i][j] = min(dp[i][j], dp[i][j+1] + 1)
res = INF
for i in range(H-K+1):
    for j in range(W-K+1):
        if max(dp[i][j], dp[i+K-1][j], dp[i][j+K-1], dp[i+K-1][j+K-1]) <= K-1:
            res = min(res, dp[i][j], dp[i+K-1][j], dp[i][j+K-1], dp[i+K-1][j+K-1])
print(-1 if res == INF else res)