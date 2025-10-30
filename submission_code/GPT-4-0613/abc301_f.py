MOD = 998244353
S = input().strip()
N = len(S)
dp = [[[0]*53 for _ in range(53)] for _ in range(N+1)]
dp[0][0][0] = 1
for i in range(N):
    for j in range(53):
        for k in range(53):
            if S[i] == '?':
                for l in range(52):
                    dp[i+1][j][k] = (dp[i+1][j][k] + dp[i][j][k]) % MOD
                    dp[i+1][l//26*26+l%26+1][k] = (dp[i+1][l//26*26+l%26+1][k] + dp[i][j][k]) % MOD
                    if j//26 == 1 and l//26 == 0:
                        dp[i+1][j][l%26+1] = (dp[i+1][j][l%26+1] + dp[i][j][k]) % MOD
                    if j//26 == 1 and l//26 == 0 and k == l%26+1:
                        dp[i+1][j][k] = (dp[i+1][j][k] + dp[i][j][k]) % MOD
            else:
                l = ord(S[i]) - ord('A') if 'A' <= S[i] <= 'Z' else ord(S[i]) - ord('a') + 26
                dp[i+1][j][k] = (dp[i+1][j][k] + dp[i][j][k]) % MOD
                dp[i+1][l//26*26+l%26+1][k] = (dp[i+1][l//26*26+l%26+1][k] + dp[i][j][k]) % MOD
                if j//26 == 1 and l//26 == 0:
                    dp[i+1][j][l%26+1] = (dp[i+1][j][l%26+1] + dp[i][j][k]) % MOD
                if j//26 == 1 and l//26 == 0 and k == l%26+1:
                    dp[i+1][j][k] = (dp[i+1][j][k] + dp[i][j][k]) % MOD
ans = pow(52, N, MOD)
for j in range(53):
    for k in range(53):
        ans = (ans - dp[N][j][k]) % MOD
print(ans)