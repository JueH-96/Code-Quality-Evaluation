# YOUR CODE HERE
import sys

def solve():
    MOD = 998244353
    N, M = map(int, input().split())
    if M == 1:
        print(0)
        return
    if N == 1:
        print(M)
        return
    dp = [0] * (N + 1)
    dp[1] = M
    dp[2] = M * (M - 1) % MOD
    for i in range(3, N + 1):
        dp[i] = ((M - 1) * (dp[i - 1] + dp[i - 2])) % MOD
    print(dp[N])

solve()