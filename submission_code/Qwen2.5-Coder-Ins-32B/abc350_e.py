# YOUR CODE HERE
def solve(N, A, X, Y):
    from functools import lru_cache

    @lru_cache(None)
    def dp(n):
        if n == 0:
            return 0.0
        cost_divide = X + dp(n // A)
        cost_roll = Y + (dp(n // 1) + dp(n // 2) + dp(n // 3) + dp(n // 4) + dp(n // 5) + dp(n // 6)) / 6
        return min(cost_divide, cost_roll)

    return dp(N)

import sys
input = sys.stdin.read().strip().split()
N = int(input[0])
A = int(input[1])
X = int(input[2])
Y = int(input[3])

print(f"{solve(N, A, X, Y):.18f}")