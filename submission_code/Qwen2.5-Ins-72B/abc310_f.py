# YOUR CODE HERE
import sys
from functools import lru_cache

MOD = 998244353

def modinv(x, mod=MOD):
    return pow(x, mod-2, mod)

def solve(N, A):
    @lru_cache(maxsize=None)
    def dp(i, s, can_be_10):
        if i == N:
            return 1 if can_be_10 else 0
        res = 0
        for j in range(1, A[i] + 1):
            new_s = s + j
            new_can_be_10 = can_be_10 or (new_s == 10)
            res += dp(i + 1, new_s % 10, new_can_be_10)
        return res % MOD

    total_ways = 1
    for a in A:
        total_ways = (total_ways * a) % MOD

    valid_ways = dp(0, 0, False)
    prob = (valid_ways * modinv(total_ways)) % MOD
    return prob

input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))
print(solve(N, A))