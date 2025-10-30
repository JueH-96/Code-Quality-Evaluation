# YOUR CODE HERE

import sys

MOD = 998244353

def solve():
    N, Q = map(int, input().split())
    P = [0] * (N + 1)
    V = [0] * (N + 1)
    for i in range(1, Q + 1):
        P[i], V[i] = map(int, input().split())
    dp = [0] * (N + 2)
    dp[0] = 1
    for i in range(1, N + 1):
        dp[i] = dp[i - 1]
        for j in range(1, Q + 1):
            if P[j] <= i:
                dp[i] = (dp[i] + dp[i - P[j]]) % MOD
    print(dp[N])

solve()