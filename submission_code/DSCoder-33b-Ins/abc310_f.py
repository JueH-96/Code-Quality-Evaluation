# YOUR CODE HERE
import sys

def solve():
    MOD = 998244353
    N = int(input())
    A = list(map(int, input().split()))
    dp = [0] * (sum(A) + 1)
    dp[0] = 1
    for a in A:
        for i in range(len(dp) - a - 1, -1, -1):
            dp[i + a] = (dp[i + a] + dp[i]) % MOD
    total = sum(dp)
    if total < total - 10:
        total += MOD
    print((total - dp[10]) % MOD)

solve()