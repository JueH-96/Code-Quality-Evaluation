def expected_cost(N, A, X, Y):
    if N == 0:
        return 0.0
    
    # Calculate the cost of the first operation
    cost_a = X + expected_cost(N // A, A, X, Y)
    
    # Calculate the expected cost of the second operation
    cost_b = Y
    for b in range(1, 7):
        cost_b += (1/6) * expected_cost(N // b, A, X, Y)
    
    # Return the minimum of the two costs
    return min(cost_a, cost_b)

import sys
input = sys.stdin.read
N, A, X, Y = map(int, input().strip().split())
result = expected_cost(N, A, X, Y)
print(f"{result:.12f}")