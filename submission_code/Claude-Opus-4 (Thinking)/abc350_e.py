import sys
from functools import lru_cache

sys.setrecursionlimit(10**6)

N, A, X, Y = map(int, input().split())

@lru_cache(maxsize=None)
def expected_cost(n):
    if n == 0:
        return 0.0
    
    # Option 1: Use operation 1
    cost1 = X + expected_cost(n // A)
    
    # Option 2: Use operation 2 (roll dice)
    # Expected cost when using dice
    sum_costs = sum(expected_cost(n // i) for i in range(2, 7))
    cost2 = (6.0 / 5.0) * Y + (1.0 / 5.0) * sum_costs
    
    return min(cost1, cost2)

result = expected_cost(N)
print(f"{result:.15f}")