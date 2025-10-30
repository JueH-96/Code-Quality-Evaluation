import sys

MOD = 998244353

def count_parenthesis(s):
    n = len(s)
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    dp[0][0] = 1

    for i in range(1, n + 1):
        if s[i - 1] == '(' or s[i - 1] == '?':
            for j in range(i + 1):
                if j > 0:
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - 1]) % MOD
                if j < i:
                    dp[i][j] = (dp[i][j] + dp[i - 1][j + 1]) % MOD
        elif s[i - 1] == ')':
            for j in range(i + 1):
                if j > 0:
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - 1]) % MOD

    return dp[n][0]

s = input().strip()
print(count_parenthesis(s))