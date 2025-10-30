import sys
from collections import defaultdict

MOD = 998244353
N = int(sys.stdin.readline())
players = [list(map(int, sys.stdin.readline().split())) for _ in range(N-1)]

fact = [1] * (N+1)
inv = [1] * (N+1)
for i in range(1, N+1):
    fact[i] = (fact[i-1] * i) % MOD
    inv[i] = pow(fact[i], MOD-2, MOD)

def comb(n, k):
    if n < k or k < 0:
        return 0
    return (fact[n] * inv[k] % MOD) * inv[n-k] % MOD

dp = [defaultdict(int) for _ in range(N+1)]
dp[0][0] = 1
for a, b in players:
    for i in range(N, a-1, -1):
        for x in dp[i-1]:
            dp[i][x+1] = (dp[i][x+1] + dp[i-1][x] * comb(x+1, a)) % MOD

ans = [0] * (N+1)
for i in range(1, N+1):
    ans[i] = sum(dp[i][j] * comb(N-1, j) for j in dp[i]) % MOD

print(' '.join(map(str, ans[1:])))