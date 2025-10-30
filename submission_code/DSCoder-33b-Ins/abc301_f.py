# YOUR CODE HERE

import sys

MOD = 998244353

def solve(s):
    q = s.count('?')
    dp = [0] * (q + 1)
    dp[0] = 1
    for i in range(q):
        dp[i + 1] = dp[i] * 52 % MOD
    ans = dp[q]
    for i in range(q):
        ans = (ans - dp[i] * 26 % MOD + MOD) % MOD
    return ans

s = sys.stdin.readline().strip()
print(solve(s))