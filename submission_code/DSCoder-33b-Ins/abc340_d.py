# YOUR CODE HERE

import sys

N = int(sys.stdin.readline())
stages = [list(map(int, sys.stdin.readline().split())) for _ in range(N-1)]

dp = [0] * N
for i in range(N-2, -1, -1):
    A, B, X = stages[i]
    dp[i] = min(A + dp[i+1], B + dp[X-1])

print(dp[0])