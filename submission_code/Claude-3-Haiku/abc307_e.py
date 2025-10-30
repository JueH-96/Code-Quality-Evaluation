# YOUR CODE HERE
N, M = map(int, input().split())
MOD = 998244353

# Dynamic programming solution
dp = [[0] * M for _ in range(N)]

# Initialize the first row
for i in range(M):
    dp[0][i] = 1

# Fill the rest of the table
for i in range(1, N):
    for j in range(M):
        for k in range(M):
            if j != k:
                dp[i][j] = (dp[i][j] + dp[i-1][k]) % MOD

# Calculate the total number of valid ways
ans = sum(dp[N-1]) % MOD
print(ans)