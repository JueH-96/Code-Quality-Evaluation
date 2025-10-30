N, W = map(int, input().split())
items = [list(map(int, input().split())) for _ in range(N)]
items.sort(key=lambda x: x[0])

dp = [[0]*(W+1) for _ in range(N+1)]
for i in range(N):
    w, v = items[i]
    for j in range(W+1):
        if j < w:
            dp[i+1][j] = dp[i][j]
        else:
            dp[i+1][j] = max(dp[i][j], dp[i+1][j-w] + v - (j//w)**2)

print(dp[N][W])