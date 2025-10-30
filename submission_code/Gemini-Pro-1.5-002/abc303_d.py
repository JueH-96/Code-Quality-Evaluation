# YOUR CODE HERE
x, y, z = map(int, input().split())
s = input()
n = len(s)
dp = [[float('inf')] * 2 for _ in range(n + 1)]
dp[0][0] = 0
for i in range(n):
    if s[i] == 'a':
        dp[i + 1][0] = min(dp[i + 1][0], dp[i][0] + x)
        dp[i + 1][1] = min(dp[i + 1][1], dp[i][1] + y)
        dp[i + 1][0] = min(dp[i + 1][0], dp[i][1] + y + z)
        dp[i + 1][1] = min(dp[i + 1][1], dp[i][0] + x + z)
    else:
        dp[i + 1][0] = min(dp[i + 1][0], dp[i][0] + y)
        dp[i + 1][1] = min(dp[i + 1][1], dp[i][1] + x)
        dp[i + 1][0] = min(dp[i + 1][0], dp[i][1] + x + z)
        dp[i + 1][1] = min(dp[i + 1][1], dp[i][0] + y + z)
print(min(dp[n][0], dp[n][1]))