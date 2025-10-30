# YOUR CODE HERE
from math import gcd

def popcount(x):
    return bin(x).count('1')

def solve(N, M):
    MOD = 998244353
    if N == 0:
        return 0
    if M == 0:
        return 0
    if N >= M:
        N, M = M, N
    cnt = bin(M).count('1')
    ans = 0
    for i in range(60):
        if M >> i & 1:
            ans += (1 << i) * (N // (1 << i) + 1) * (N // (1 << i) + 2) // 2 % MOD
            ans %= MOD
    return ans

N, M = map(int, input().split())
print(solve(N, M))