import sys
from collections import defaultdict
from operator import mul
from functools import reduce

MOD = 998244353

def comb(n, r):
    if r > n - r:
        r = n - r
    if r == 0:
        return 1
    if r == 1:
        return n
    numerator = [n - r + k + 1 for k in range(r)]
    denominator = [k + 1 for k in range(r)]
    for p in range(2, r+1):
        pivot = denominator[p - 1]
        if pivot > 1:
            offset = (n - r) % p
            for k in range(p-1, r, p):
                numerator[k - offset] /= pivot
                denominator[k] /= pivot
    result = 1
    for k in range(r):
        if numerator[k] > 1:
            result *= int(numerator[k])
            result %= MOD
    return result

def solve():
    N, M, K = map(int, sys.stdin.readline().split())
    edges = defaultdict(list)
    for _ in range(M):
        X, Y = map(int, sys.stdin.readline().split())
        edges[X].append(Y)
    dp = [[0] * (N+1) for _ in range(K+1)]
    dp[0][1] = 1
    for k in range(K):
        total = sum(dp[k]) % MOD
        for i in range(1, N+1):
            dp[k+1][i] = (total - dp[k][i] + MOD) % MOD
            for j in edges[i]:
                dp[k+1][j] = (dp[k+1][j] + dp[k][i]) % MOD
    print(sum(dp[K]) % MOD)

solve()