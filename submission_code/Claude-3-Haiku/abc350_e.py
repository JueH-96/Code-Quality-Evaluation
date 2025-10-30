import math
import random

def solve(N, A, X, Y):
    # Initialize the expected cost to 0
    expected_cost = 0

    # Iterate until N becomes 0
    while N > 0:
        # Option 1: Pay X yen to replace N with floor(N/A)
        cost1 = X
        new_N = math.floor(N / A)
        expected_cost_1 = cost1 + solve(new_N, A, X, Y)

        # Option 2: Pay Y yen to roll a die and replace N with floor(N/b)
        cost2 = Y
        b = random.randint(1, 6)
        new_N = math.floor(N / b)
        expected_cost_2 = cost2 + solve(new_N, A, X, Y)

        # Choose the option with the minimum expected cost
        expected_cost += min(expected_cost_1, expected_cost_2)
        N = new_N

    return expected_cost