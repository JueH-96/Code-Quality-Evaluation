# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

K = int(data[0])
C = list(map(int, data[1:]))

MOD = 998244353

dp = [[0] * (K + 1) for _ in range(27)]
dp[0][0] = 1

for i in range(1, 27):
    for j in range(K + 1):
        dp[i][j] = dp[i - 1][j]
        if j >= 1:
            dp[i][j] += dp[i][j - 1]
            dp[i][j] %= MOD
        if j >= C[i - 1] + 1:
            dp[i][j] -= dp[i - 1][j - C[i - 1] - 1]
            dp[i][j] %= MOD

print(dp[26][K])