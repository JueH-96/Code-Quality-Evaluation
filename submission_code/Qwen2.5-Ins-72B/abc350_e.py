# YOUR CODE HERE
import sys

def min_expected_cost(N, A, X, Y):
    if N == 0:
        return 0.0

    # Precompute the expected cost for N = 1 to 6
    expected_cost = [0.0] * 7
    for i in range(1, 7):
        expected_cost[i] = Y + (1.0 / 6) * sum(expected_cost[j] for j in range(1, 7) if j <= i)

    # Dynamic programming to compute the expected cost for larger N
    dp = [0.0] * (N + 1)
    for i in range(1, N + 1):
        if i < 7:
            dp[i] = expected_cost[i]
        else:
            cost_div_A = X + dp[i // A]
            cost_die = Y + (1.0 / 6) * sum(dp[i // b] for b in range(1, 7))
            dp[i] = min(cost_div_A, cost_die)

    return dp[N]

# Read input
N, A, X, Y = map(int, sys.stdin.readline().strip().split())

# Compute and print the result
print(min_expected_cost(N, A, X, Y))