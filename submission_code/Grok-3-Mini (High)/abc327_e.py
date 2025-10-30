import sys
import math

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
P = list(map(int, data[index:index + N]))

# Initialize DP table
dp = [[float('-inf') for _ in range(N + 1)] for _ in range(N + 1)]
dp[0][0] = 0.0

# Fill DP table
for i in range(1, N + 1):
    dp[i][0] = 0.0  # No contests chosen
    for k in range(1, N + 1):
        val_not_include = dp[i - 1][k]
        val_include = 0.9 * dp[i - 1][k - 1] + P[i - 1]
        dp[i][k] = max(val_not_include, val_include)

# Compute the maximum rating
max_R = float('-inf')
for k in range(1, N + 1):
    S_k = dp[N][k]  # Maximum weighted sum for size k
    D_k = 10 * (1 - 0.9 ** k)  # Denominator for size k
    A_k = S_k / D_k  # Weighted average
    R_k = A_k - 1200 / math.sqrt(k)  # Rating
    if R_k > max_R:
        max_R = R_k

# Output the result with high precision
print(f"{max_R:.15f}")