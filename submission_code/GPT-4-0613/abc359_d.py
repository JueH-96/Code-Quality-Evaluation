MOD = 998244353
N, K = map(int, input().split())
S = input()

dp = [[[0]*2 for _ in range(K+1)] for _ in range(N+1)]
dp[0][0][1] = 1
dp[0][0][0] = 1

for i in range(N):
    for j in range(K):
        for k in range(2):
            if S[i] != 'B':
                dp[i+1][j+1][k|(j==K-1)] += dp[i][j][k]
                dp[i+1][j+1][k|(j==K-1)] %= MOD
                dp[i+1][1][k|(j==K-1)] += dp[i][j][k]
                dp[i+1][1][k|(j==K-1)] %= MOD
            if S[i] != 'A':
                dp[i+1][1][k|(j==K-1)] += dp[i][j][k]
                dp[i+1][1][k|(j==K-1)] %= MOD
                dp[i+1][j+1][k|(j==K-1)] += dp[i][j][k]
                dp[i+1][j+1][k|(j==K-1)] %= MOD
    for k in range(2):
        dp[i+1][0][k] += dp[i][K][k]
        dp[i+1][0][k] %= MOD
        dp[i+1][0][k] += dp[i][K-1][k]
        dp[i+1][0][k] %= MOD

print((pow(2, N, MOD) - dp[N][0][0] - dp[N][K][0] + MOD + MOD) % MOD)