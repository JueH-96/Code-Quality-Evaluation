MOD = 998244353

def solve():
    S = input().strip()
    n = len(S)
    dp = [[0]*27 for _ in range(n+1)]
    dp[0][0] = 1
    for i in range(n):
        if S[i] == '?':
            for j in range(27):
                dp[i+1][j] = (dp[i+1][j] + dp[i][j]*26) % MOD
                if j < 26:
                    dp[i+1][j+1] = (dp[i+1][j+1] + dp[i][j]*25) % MOD
        else:
            c = ord(S[i]) - ord('A')
            for j in range(27):
                dp[i+1][j] = (dp[i+1][j] + dp[i][j]*(26 if j != c else 25)) % MOD
                if j < 26:
                    dp[i+1][j+1] = (dp[i+1][j+1] + dp[i][j]*(25 if j != c else 24)) % MOD
    print(dp[n][26])

solve()