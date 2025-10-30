import sys

MOD = 998244353
N, K = map(int, sys.stdin.readline().split())
S = sys.stdin.readline().strip()

dp = [[0]*2 for _ in range(N+1)]
dp[0][0] = 1

for i in range(N):
    if S[i] == 'A' or S[i] == '?':
        dp[i+1][0] = (dp[i+1][0] + dp[i][0]) % MOD
        dp[i+1][1] = (dp[i+1][1] + dp[i][0]) % MOD
    if S[i] == 'B' or S[i] == '?':
        dp[i+1][0] = (dp[i+1][0] + dp[i][1]) % MOD
        dp[i+1][1] = (dp[i+1][1] + dp[i][1]) % MOD

    if i >= K-1:
        dp[i+1][0] = (dp[i+1][0] - dp[i-K+1][0] + MOD) % MOD
        dp[i+1][1] = (dp[i+1][1] - dp[i-K+1][1] + MOD) % MOD

print(dp[N][0])