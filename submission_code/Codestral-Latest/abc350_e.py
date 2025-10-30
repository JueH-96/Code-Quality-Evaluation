import sys
import math

def expected_cost(N, A, X, Y):
    # Base case: if N is 0, no cost is needed
    if N == 0:
        return 0

    # Expected cost for the first operation
    cost1 = X + expected_cost(math.floor(N / A), A, X, Y)

    # Expected cost for the second operation
    cost2 = Y
    for b in range(1, 7):
        cost2 += expected_cost(math.floor(N / b), A, X, Y) / 6

    # Return the minimum of the two costs
    return min(cost1, cost2)

# Read input
N, A, X, Y = map(int, sys.stdin.read().split())

# Calculate and print the expected minimum cost
print("{:.15f}".format(expected_cost(N, A, X, Y)))