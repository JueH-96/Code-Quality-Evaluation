# YOUR CODE HERE
import sys
from functools import lru_cache

MOD = 998244353

def solve(K, C):
    @lru_cache(maxsize=None)
    def dp(n, i):
        if n == 0:
            return 1
        if i == 26:
            return 0
        result = 0
        for j in range(min(n, C[i]) + 1):
            result += dp(n - j, i + 1)
            result %= MOD
        return result

    total = 0
    for n in range(1, K + 1):
        total += dp(n, 0)
        total %= MOD
    return total

K = int(input())
C = list(map(int, input().split()))
print(solve(K, C))