# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

MOD = 998244353

# Initialize DP table
dp = [[0] * (1001) for _ in range(N + 1)]
dp[0][0] = 1

# Fill DP table
for i in range(1, N + 1):
    for j in range(1001):
        dp[i][j] = dp[i - 1][j]
        if j >= A[i - 1]:
            dp[i][j] += dp[i - 1][j - A[i - 1]]
        dp[i][j] %= MOD

# Calculate the probability
result = 0
for i in range(1, N + 1):
    result += dp[i][10]
    result %= MOD

print(result)