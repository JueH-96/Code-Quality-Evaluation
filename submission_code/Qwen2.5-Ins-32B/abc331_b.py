import sys

def min_cost_eggs(N, S, M, L):
    # Initialize the dp array with infinity values
    dp = [float('inf')] * (N + 1)
    dp[0] = 0  # Base case: cost to buy 0 eggs is 0

    # Fill the dp array
    for i in range(1, N + 1):
        if i >= 6:
            dp[i] = min(dp[i], dp[i - 6] + S)
        if i >= 8:
            dp[i] = min(dp[i], dp[i - 8] + M)
        if i >= 12:
            dp[i] = min(dp[i], dp[i - 12] + L)

    # Find the minimum cost to buy at least N eggs
    return min(dp[N:])

# Read input from stdin
N, S, M, L = map(int, sys.stdin.readline().split())

# Calculate and print the answer
print(min_cost_eggs(N, S, M, L))