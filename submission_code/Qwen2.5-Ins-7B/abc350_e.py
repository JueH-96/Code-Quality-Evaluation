# YOUR CODE HERE
from math import inf
from fractions import Fraction

def expected_cost(N, A, X, Y):
    memo = {}
    
    def dp(n):
        if n == 0:
            return 0
        if n in memo:
            return memo[n]
        
        cost = inf
        # Option 1: Divide by A
        cost = min(cost, X + dp(n // A))
        
        # Option 2: Roll a die
        for b in range(1, 7):
            cost = min(cost, Y + dp(n // b))
        
        memo[n] = cost
        return cost
    
    return dp(N)

N, A, X, Y = map(int, input().split())
print(expected_cost(N, A, X, Y))