MOD = 998244353

def count_no_ddos(s):
    n = len(s)
    dp = [[0] * 4 for _ in range(n + 1)]
    dp[0][0] = 1

    for i in range(1, n + 1):
        if s[i - 1] == '?':
            dp[i][0] = (dp[i - 1][0] * 52) % MOD
            dp[i][1] = (dp[i - 1][1] * 26) % MOD
            dp[i][2] = (dp[i - 1][2] * 26) % MOD
            dp[i][3] = (dp[i - 1][3] * 26) % MOD
            if i >= 2 and dp[i - 2][1] != 0:
                dp[i][2] = (dp[i][2] + dp[i - 2][1] * 26) % MOD
            if i >= 3 and dp[i - 3][2] != 0:
                dp[i][3] = (dp[i][3] + dp[i - 3][2] * 26) % MOD
        elif s[i - 1].isupper():
            dp[i][0] = dp[i - 1][0]
            dp[i][1] = dp[i - 1][1] + dp[i - 1][0]
            dp[i][2] = dp[i - 1][2]
            dp[i][3] = dp[i - 1][3]
            if i >= 2 and dp[i - 2][1] != 0:
                dp[i][2] = (dp[i][2] + dp[i - 2][1]) % MOD
            if i >= 3 and dp[i - 3][2] != 0:
                dp[i][3] = (dp[i][3] + dp[i - 3][2]) % MOD
        else:
            dp[i][0] = dp[i - 1][0]
            dp[i][1] = dp[i - 1][1]
            dp[i][2] = dp[i - 1][2] + dp[i - 1][1]
            dp[i][3] = dp[i - 1][3]

    return dp[n][0]

s = input()
print(count_no_ddos(s))