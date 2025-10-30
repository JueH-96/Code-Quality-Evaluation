import sys

MOD = 998244353

def count_valid_parentheses(S):
    n = len(S)
    # dp[i][j][k] will store the number of valid parenthesis strings
    # from S[0] to S[i] with j open parentheses and k question marks resolved
    dp = [[[0] * (n + 1) for _ in range(n + 1)] for _ in range(n + 1)]
    dp[0][0][0] = 1

    for i in range(n):
        for j in range(n):
            for k in range(n):
                if dp[i][j][k] == 0:
                    continue
                if S[i] == '(' or S[i] == '?':
                    dp[i + 1][j + 1][k] = (dp[i + 1][j + 1][k] + dp[i][j][k]) % MOD
                if S[i] == ')' or S[i] == '?':
                    if j > 0:
                        dp[i + 1][j - 1][k + (S[i] == '?')] = (dp[i + 1][j - 1][k + (S[i] == '?')] + dp[i][j][k]) % MOD

    total_valid = 0
    for j in range(n + 1):
        for k in range(n + 1):
            if j == 0:
                total_valid = (total_valid + dp[n][j][k]) % MOD

    return total_valid

if __name__ == "__main__":
    input = sys.stdin.read().strip()
    result = count_valid_parentheses(input)
    sys.stdout.write(str(result) + '
')