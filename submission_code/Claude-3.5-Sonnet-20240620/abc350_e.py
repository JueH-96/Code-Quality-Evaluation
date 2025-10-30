# YOUR CODE HERE
import sys
import math

def solve(N, A, X, Y):
    dp = [0] * (N + 1)
    
    for i in range(1, N + 1):
        # Option 1: Pay X yen
        cost1 = X + dp[i // A]
        
        # Option 2: Pay Y yen and roll a die
        cost2 = Y
        for j in range(1, 7):
            cost2 += dp[i // j] / 6
        
        dp[i] = min(cost1, cost2)
    
    return dp[N]

N, A, X, Y = map(int, sys.stdin.readline().split())
result = solve(N, A, X, Y)
print(f"{result:.15f}")