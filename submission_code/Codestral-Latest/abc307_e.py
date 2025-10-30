# YOUR CODE HERE
MOD = 998244353

def count_ways(N, M):
    if N == 1:
        return M
    if N == 2:
        return M * (M - 1)

    # Initialize the DP table
    dp = [[0] * M for _ in range(N)]

    # Base case: for N = 1, each person can have any of M values
    for j in range(M):
        dp[0][j] = 1

    # Fill the DP table
    for i in range(1, N):
        for j in range(M):
            for k in range(M):
                if j != k:
                    dp[i][j] = (dp[i][j] + dp[i - 1][k]) % MOD

    # Sum up the results for the last person
    total_ways = 0
    for j in range(M):
        total_ways = (total_ways + dp[N - 1][j]) % MOD

    return total_ways

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

# Calculate and print the result
result = count_ways(N, M)
print(result)