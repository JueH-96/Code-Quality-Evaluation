import sys

MOD = 998244353
N = int(sys.stdin.readline().strip())
A = list(map(int, sys.stdin.readline().strip().split()))

fact = [0] * (N + 1)
factinv = [0] * (N + 1)
fact[0] = factinv[0] = 1
for i in range(1, N + 1):
    fact[i] = (fact[i - 1] * i) % MOD
    factinv[i] = factinv[i - 1] * pow(i, MOD - 2, MOD)

dp = [0] * (N + 1)
dp[0] = 1
for i in range(1, N + 1):
    dp[i] = (dp[i - 1] * i) % MOD

A.sort()

ans = 0
for i in range(1, N + 1):
    ans = (ans + A[i - 1] * (dp[i] - dp[i - 1]) % MOD * factinv[i]) % MOD

print(ans)