# YOUR CODE HERE
MOD = 998244353

def count_valid_parentheses(s):
    n = len(s)
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    dp[0][0] = 1

    for i in range(1, n + 1):
        for j in range(n + 1):
            if s[i - 1] == '(' or s[i - 1] == '?':
                dp[i][j + 1] = (dp[i][j + 1] + dp[i - 1][j]) % MOD
            if j > 0 and (s[i - 1] == ')' or s[i - 1] == '?'):
                dp[i][j - 1] = (dp[i][j - 1] + dp[i - 1][j]) % MOD

    return dp[n][0]

s = input().strip()
result = count_valid_parentheses(s)
print(result)