import sys
from collections import defaultdict

MOD = 998244353
N, K = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))
S = [0]*(N+1)
for i in range(N):
    S[i+1] = S[i] + A[i]

dp = [0]*(N+1)
dp[0] = 1
sum_dp = [0]*(N+2)
sum_dp[1] = 1
last = defaultdict(int)
for i in range(1, N+1):
    dp[i] = sum_dp[i]
    if S[i] - K in last:
        dp[i] -= sum_dp[last[S[i] - K]]
        dp[i] %= MOD
    sum_dp[i+1] = sum_dp[i] + dp[i]
    sum_dp[i+1] %= MOD
    last[S[i]] = i + 1

print(dp[N])