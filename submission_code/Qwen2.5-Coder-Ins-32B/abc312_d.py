# YOUR CODE HERE
MOD = 998244353

def count_parenthesis_ways(S):
    n = len(S)
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    dp[0][0] = 1

    for i in range(n):
        for j in range(n + 1):
            if S[i] == '(' or S[i] == '?':
                if j + 1 <= n:
                    dp[i + 1][j + 1] = (dp[i + 1][j + 1] + dp[i][j]) % MOD
            if S[i] == ')' or S[i] == '?':
                if j - 1 >= 0:
                    dp[i + 1][j - 1] = (dp[i + 1][j - 1] + dp[i][j]) % MOD

    return dp[n][0]

import sys
S = sys.stdin.read().strip()
print(count_parenthesis_ways(S))