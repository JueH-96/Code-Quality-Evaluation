import sys
sys.setrecursionlimit(1000)

N, A, X, Y = map(int, input().split())

memo = {}

def dp(n):
    if n == 0:
        return 0.0
    if n in memo:
        return memo[n]
    
    cost1 = X + dp(n // A)
    cost2 = (6.0 * Y + sum(dp(n // b) for b in range(2, 7))) / 5.0
    result = min(cost1, cost2)
    memo[n] = result
    return result

print(dp(N))