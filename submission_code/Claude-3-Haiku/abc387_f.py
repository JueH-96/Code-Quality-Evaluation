# Read the input
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Initialize a 2D array to store the number of valid sequences
dp = [[0] * (M + 1) for _ in range(N + 1)]

# Base case: there is 1 valid sequence when N = 0
for j in range(1, M + 1):
    dp[0][j] = 1

# Compute the number of valid sequences
for i in range(1, N + 1):
    for j in range(1, M + 1):
        if j <= A[i - 1]:
            dp[i][j] = (dp[i - 1][j] + dp[i][j - 1]) % 998244353
        else:
            dp[i][j] = dp[i][j - 1]

# Print the answer
print(dp[N][M])