MOD = 998244353

def count_valid_strings(S):
    n = len(S)
    dp = [[0, 0, 0, 0] for _ in range(n + 1)]
    dp[0][0] = 1

    for i in range(n):
        for j in range(4):
            if S[i] == '?':
                if j == 0:
                    dp[i + 1][j] = (dp[i][j] * 26) % MOD
                elif j == 1:
                    dp[i + 1][j] = (dp[i][j] * 26 + dp[i][0] * 25) % MOD
                elif j == 2:
                    dp[i + 1][j] = (dp[i][j] * 26 + dp[i][1] * 25) % MOD
                elif j == 3:
                    dp[i + 1][j] = (dp[i][j] * 26 + dp[i][2] * 25) % MOD
            elif S[i].isupper():
                if j == 0:
                    dp[i + 1][j] = dp[i][j]
                elif j == 1:
                    dp[i + 1][j] = (dp[i][j] + dp[i][0]) % MOD
                elif j == 2:
                    dp[i + 1][j] = (dp[i][j] + dp[i][1]) % MOD
                elif j == 3:
                    dp[i + 1][j] = (dp[i][j] + dp[i][2]) % MOD
            else:
                if j == 0:
                    dp[i + 1][j] = dp[i][j]
                elif j == 1:
                    dp[i + 1][j] = dp[i][j]
                elif j == 2:
                    dp[i + 1][j] = dp[i][j]
                elif j == 3:
                    dp[i + 1][j] = dp[i][j]

    return (dp[n][0] + dp[n][1] + dp[n][2] + dp[n][3]) % MOD

# Read input
S = input().strip()

# Calculate and print the result
print(count_valid_strings(S))