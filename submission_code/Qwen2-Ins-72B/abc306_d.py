n = int(input())
dp = [[0, 0] for _ in range(n+1)]
for i in range(n):
    x, y = map(int, input().split())
    if x == 0:
        dp[i+1][0] = max(dp[i][0], dp[i][1]) + y
        dp[i+1][1] = max(dp[i][0], dp[i][1])
    else:
        dp[i+1][0] = dp[i][1]
        dp[i+1][1] = max(dp[i][0], dp[i][1]) + y
print(max(dp[n][0], dp[n][1]))