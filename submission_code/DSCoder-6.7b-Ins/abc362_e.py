import sys

MOD = 998244353
N = int(sys.stdin.readline().strip())
A = list(map(int, sys.stdin.readline().strip().split()))

dp = [[0]*N for _ in range(N)]
cnt = [[0]*N for _ in range(N)]

for i in range(N):
    dp[i][i] = 1
    cnt[i][i] = 1

for i in range(N-1, -1, -1):
    for j in range(i+1, N):
        if A[j] - A[i] <= 1:
            dp[i][j] = (dp[i][j-1] + dp[i+1][j] + 1) % MOD
            cnt[i][j] = (cnt[i][j-1] + cnt[i+1][j] - cnt[i+1][j-1] + dp[i+1][j] + 1) % MOD
        else:
            dp[i][j] = (dp[i][j-1] + dp[i+1][j] - dp[i+1][j-1]) % MOD
            cnt[i][j] = (cnt[i][j-1] + cnt[i+1][j] - cnt[i+1][j-1]) % MOD

for i in range(N):
    print(cnt[i][i], end=' ')