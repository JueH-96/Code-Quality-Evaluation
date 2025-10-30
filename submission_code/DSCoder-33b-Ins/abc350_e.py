# YOUR CODE HERE
import sys

def solve(N, A, X, Y):
    dp = [0] * (N + 1)
    for i in range(2, N + 1):
        dp[i] = min(dp[i - 1] + X, min(dp[i // A] + X + (i % A) * Y, dp[i // A + 1] * A + X * (A - (i % A)) + Y * (i % A)))
    return dp[N]

N, A, X, Y = map(int, sys.stdin.readline().split())
print(solve(N, A, X, Y))