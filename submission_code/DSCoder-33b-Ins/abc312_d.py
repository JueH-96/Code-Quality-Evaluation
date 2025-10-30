# YOUR CODE HERE

import sys

MOD = 998244353

def solve(s):
    n = len(s)
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    for i in range(n):
        for j in range(i + 1):
            if s[i] == '(' or s[i] == '?':
                dp[i + 1][j + 1] += dp[i][j]
                dp[i + 1][j + 1] %= MOD
            if s[i] == ')' or s[i] == '?':
                if j > 0:
                    dp[i + 1][j - 1] += dp[i][j]
                    dp[i + 1][j - 1] %= MOD
    return dp[n][0]

s = sys.stdin.readline().strip()
print(solve(s))