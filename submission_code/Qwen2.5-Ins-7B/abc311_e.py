# YOUR CODE HERE
h, w, n = map(int, input().split())
holes = [tuple(map(int, input().split())) for _ in range(n)]
dp = [[0] * (w + 1) for _ in range(h + 1)]
for a, b in holes:
    dp[a][b] = -1
ans = 0
for i in range(1, h + 1):
    for j in range(1, w + 1):
        if dp[i][j] == -1:
            continue
        dp[i][j] = min(dp[i][j], dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
        ans += dp[i][j]
print(ans)