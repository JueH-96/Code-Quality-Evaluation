import sys

def expected_cost(N, A, X, Y):
    # Dynamic programming table to store the minimum expected cost for each value of N
    dp = [0.0] * (N + 1)

    # Iterate backwards from N to 1
    for i in range(N, 0, -1):
        # Option 1: Pay X yen to replace N with floor(N/A)
        cost1 = X + dp[i // A]

        # Option 2: Pay Y yen to roll a die and replace N with floor(N/b)
        cost2 = Y
        for b in range(1, 7):
            cost2 += dp[i // b] / 6

        # Choose the option with the minimum expected cost
        dp[i] = min(cost1, cost2)

    return dp[N]

# Read input from stdin
input = sys.stdin.read()
N, A, X, Y = map(int, input.split())

# Calculate and print the minimum expected cost
print(expected_cost(N, A, X, Y))