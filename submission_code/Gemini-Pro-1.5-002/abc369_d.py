# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

dp = [[0] * (n + 1) for _ in range(n + 1)]

for i in range(n - 1, -1, -1):
    for j in range(i + 1):
        if (i + 1 - j) % 2 == 1:
            dp[i][j] = max(dp[i + 1][j + 1] + a[i], dp[i + 1][j])
        else:
            dp[i][j] = max(dp[i + 1][j + 1] + 2 * a[i], dp[i + 1][j])

print(dp[0][0])