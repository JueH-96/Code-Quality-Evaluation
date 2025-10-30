import sys
N, M = map(int, sys.stdin.readline().split())
mod = 998244353
dp = [0] * (N+1)
dp[0] = 1
dp[1] = M
for i in range(2, N+1):
    dp[i] = (dp[i-1] * (M-1) + dp[i-2] * (M-1) * (M-2)) % mod
print(dp[N])