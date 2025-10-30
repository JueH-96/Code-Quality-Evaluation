import sys
from collections import defaultdict

def solve():
    N, M = map(int, input().split())
    MOD = 998244353
    conditions = defaultdict(list)
    for _ in range(M):
        L, R, X = map(int, input().split())
        conditions[X-1].append((L-1, R-1))
    
    dp = [0] * (N + 1)
    dp[0] = 1
    for i in range(1, N + 1):
        dp[i] = dp[i-1] * i % MOD
    
    inv = [0] * (N + 1)
    inv[1] = 1
    for i in range(2, N + 1):
        inv[i] = MOD - MOD // i * inv[MOD % i] % MOD
    
    for x in range(N):
        if x not in conditions:
            continue
        for L, R in conditions[x]:
            if L == R:
                print(0)
                return
            for r in range(R, N):
                dp[r+1] -= dp[L] * dp[r-L] * inv[R-L+1] % MOD
                dp[r+1] %= MOD
    
    print(dp[N])

solve()