MOD = 998244353

def count_parenthesis_ways(s):
    n = len(s)
    # dp[i][j] will store the number of ways to form a valid parenthesis string
    # from the first i characters with j open parentheses
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    dp[0][0] = 1

    for i in range(1, n + 1):
        for j in range(n + 1):
            if s[i - 1] == '(':
                if j > 0:
                    dp[i][j] = dp[i - 1][j - 1]
            elif s[i - 1] == ')':
                if j < n:
                    dp[i][j] = dp[i - 1][j + 1]
            else:  # s[i - 1] == '?'
                if j > 0:
                    dp[i][j] += dp[i - 1][j - 1]
                if j < n:
                    dp[i][j] += dp[i - 1][j + 1]
            dp[i][j] %= MOD

    return dp[n][0]

# Read input from stdin
S = input().strip()

# Calculate and print the result
print(count_parenthesis_ways(S))