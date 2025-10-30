import sys
import random

def solve(N, A, X, Y):
    dp = [0] * (N + 1)
    dp[0] = dp[1] = 0
    for i in range(2, N + 1):
        dp[i] = min(dp[i//A] + X, dp[random.randint(1, i//2)] + Y)
    return dp[N]

N, A, X, Y = map(int, sys.stdin.readline().split())
print(solve(N, A, X, Y))