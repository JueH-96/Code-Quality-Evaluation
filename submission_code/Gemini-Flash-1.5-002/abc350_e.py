# YOUR CODE HERE
import sys

def solve():
    n, a, x, y = map(int, sys.stdin.readline().split())
    
    dp = {}
    def dfs(n):
        if n == 0:
            return 0
        if n in dp:
            return dp[n]
        
        cost1 = x + dfs(n // a)
        cost2 = y + sum(dfs(n // i) for i in range(1, 7)) / 6.0
        
        dp[n] = min(cost1, cost2)
        return dp[n]
    
    print(dfs(n))

solve()