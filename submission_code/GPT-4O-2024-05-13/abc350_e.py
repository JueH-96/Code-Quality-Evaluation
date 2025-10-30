# YOUR CODE HERE
import math

def min_expected_cost(N, A, X, Y):
    # Memoization dictionary to store the minimum expected cost for each value of N
    memo = {}

    def expected_cost(n):
        if n == 0:
            return 0
        if n in memo:
            return memo[n]

        # Cost of using the first operation
        cost1 = X + expected_cost(n // A)

        # Expected cost of using the second operation
        cost2 = Y + sum(expected_cost(n // b) for b in range(1, 7)) / 6

        # Minimum cost
        result = min(cost1, cost2)
        memo[n] = result
        return result

    return expected_cost(N)

# Read input
import sys
input = sys.stdin.read
data = input().strip().split()
N = int(data[0])
A = int(data[1])
X = int(data[2])
Y = int(data[3])

# Calculate and print the result
result = min_expected_cost(N, A, X, Y)
print(f"{result:.15f}")