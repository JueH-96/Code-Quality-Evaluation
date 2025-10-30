import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
K = int(data[2])
X = list(map(int, data[3:]))

MOD = 998244353

# Initialize a DP table
dp = [[0] * (M + 1) for _ in range(N + 1)]

# Base case: no elements in sequence A
for j in range(M + 1):
    dp[0][j] = 1

# Fill the DP table
for i in range(1, N + 1):
    for j in range(M + 1):
        dp[i][j] = dp[i - 1][j] * K % MOD
        if j > 0:
            dp[i][j] -= dp[i - 1][j - 1]
            dp[i][j] %= MOD

# Calculate the result
result = 0
for j in range(M + 1):
    result += dp[N][j] * (K - 1) ** (M - j)
    result %= MOD

print(result)