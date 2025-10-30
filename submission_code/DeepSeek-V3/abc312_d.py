# YOUR CODE HERE
MOD = 998244353

def count_valid_parentheses(S):
    n = len(S)
    # dp[i][j] represents the number of ways to form a valid parentheses string
    # using the first i characters of S, with j being the current balance (number of '(' minus number of ')')
    dp = [[0] * (n + 2) for _ in range(n + 1)]
    dp[0][0] = 1
    
    for i in range(n):
        for j in range(n + 1):
            if dp[i][j] == 0:
                continue
            if S[i] == '(':
                if j + 1 <= n:
                    dp[i+1][j+1] = (dp[i+1][j+1] + dp[i][j]) % MOD
            elif S[i] == ')':
                if j - 1 >= 0:
                    dp[i+1][j-1] = (dp[i+1][j-1] + dp[i][j]) % MOD
            else:  # S[i] == '?'
                if j + 1 <= n:
                    dp[i+1][j+1] = (dp[i+1][j+1] + dp[i][j]) % MOD
                if j - 1 >= 0:
                    dp[i+1][j-1] = (dp[i+1][j-1] + dp[i][j]) % MOD
    
    return dp[n][0]

# Read input
S = input().strip()
# Compute and print the result
print(count_valid_parentheses(S))