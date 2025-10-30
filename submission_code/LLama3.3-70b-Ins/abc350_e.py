import sys
from collections import defaultdict

def calculate_min_cost(n, a, x, y):
    """
    Calculate the minimum expected cost to reduce n to 0.

    Args:
    n (int): The initial number.
    a (int): The divisor for the first operation.
    x (int): The cost of the first operation.
    y (int): The cost of the second operation.

    Returns:
    float: The minimum expected cost.
    """
    memo = defaultdict(lambda: float('inf'))

    def dp(n):
        # Base case: If n is 0, the cost is 0.
        if n == 0:
            return 0

        # If the result is already calculated, return it.
        if n in memo:
            return memo[n]

        # Calculate the cost of the first operation.
        cost_a = x + dp(n // a)

        # Calculate the expected cost of the second operation.
        expected_cost_b = y
        for b in range(1, 7):
            expected_cost_b += (1/6) * dp(n // b)

        # Choose the operation with the minimum cost.
        memo[n] = min(cost_a, expected_cost_b)

        return memo[n]

    return dp(n)

# Read the inputs from stdin.
n, a, x, y = map(int, sys.stdin.readline().split())

# Calculate and print the minimum expected cost.
min_cost = calculate_min_cost(n, a, x, y)
print(f"{min_cost:.10f}")