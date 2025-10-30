def count_parenthesis_ways(S):
    MOD = 998244353
    n = len(S)
    
    # dp[i][j] will store the number of valid parenthesis strings that can be formed
    # using the first i characters of S with j open parentheses
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    dp[0][0] = 1  # Base case: empty string can form a valid parenthesis string

    for i in range(n):
        for j in range(n):
            if S[i] == '(' or S[i] == '?':
                dp[i + 1][j + 1] = (dp[i + 1][j + 1] + dp[i][j]) % MOD
            if S[i] == ')' or S[i] == '?':
                if j > 0:
                    dp[i + 1][j - 1] = (dp[i + 1][j - 1] + dp[i][j]) % MOD
            if S[i] == '?':
                dp[i + 1][j] = (dp[i + 1][j] + dp[i][j] * 2) % MOD

    # The answer is the number of ways to have 0 open parentheses after processing all characters
    return dp[n][0]

import sys
input = sys.stdin.read

if __name__ == "__main__":
    S = input().strip()
    print(count_parenthesis_ways(S))