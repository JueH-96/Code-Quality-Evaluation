import sys
from operator import itemgetter
from bisect import bisect_right
from itertools import accumulate

MOD = 998244353
N, M = map(int, sys.stdin.readline().split())
LRX = sorted([tuple(map(int, sys.stdin.readline().split())) for _ in range(M)], key=itemgetter(1))
fact = [1]*(N+1)
for i in range(1, N+1):
    fact[i] = fact[i-1]*i%MOD
inv = [1]*(N+1)
inv[N] = pow(fact[N], MOD-2, MOD)
for i in range(N, 0, -1):
    inv[i-1] = inv[i]*i%MOD
dp = [0]*(N+1)
dp[0] = 1
cum = [1]*(N+1)
idx = 0
for r in range(1, N+1):
    while idx < M and LRX[idx][1] == r:
        l, x = LRX[idx][0], LRX[idx][2]
        if l <= x-1:
            dp[r] -= dp[x-1]*cum[l-1]*inv[r-l]%MOD
        if x+1 <= r:
            dp[r] -= dp[r-1]*cum[x]%MOD
        idx += 1
    dp[r] %= MOD
    cum[r] = (cum[r-1]+dp[r]*inv[r])%MOD
print(dp[N])