import sys

def min_cost(N, A, X, Y):
    memo = {}
    
    def dp(n):
        if n in memo:
            return memo[n]
        if n == 0:
            return 0
        cost_replace = X + dp(n // A)
        cost_roll = Y + (1/6) * sum(dp(n // b) for b in range(1, 7))
        memo[n] = min(cost_replace, cost_roll)
        return memo[n]
    
    return dp(N)

# Read input from stdin
N, A, X, Y = map(int, input().split())

# Calculate and print the result
print(min_cost(N, A, X, Y))