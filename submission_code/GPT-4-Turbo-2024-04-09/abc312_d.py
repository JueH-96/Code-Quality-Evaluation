MOD = 998244353

def count_valid_parentheses(s):
    n = len(s)
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    
    # Base case: empty substring is a valid parenthesis string
    for i in range(n + 1):
        dp[i][i] = 1
    
    # Fill dp table
    for length in range(2, n + 1, 2):  # Only even lengths can be valid parenthesis strings
        for i in range(n - length + 1):
            j = i + length
            # dp[i][j] is the number of valid parenthesis strings from s[i:j]
            # Consider s[i] paired with some k in [i+1, j]
            for k in range(i + 1, j + 1, 2):
                if s[i] in '([' and s[k] in ')]':
                    left_pair = 1 if s[i] == '(' or s[k] == ')' else 2
                    dp[i][j] = (dp[i][j] + left_pair * dp[i + 1][k] * dp[k + 1][j]) % MOD
                elif s[i] == '?' and s[k] == '?':
                    dp[i][j] = (dp[i][j] + 2 * dp[i + 1][k] * dp[k + 1][j]) % MOD
    
    return dp[0][n]

import sys
input = sys.stdin.read
s = input().strip()
print(count_valid_parentheses(s))