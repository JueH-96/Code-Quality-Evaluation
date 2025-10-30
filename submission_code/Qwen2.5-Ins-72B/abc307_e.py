# YOUR CODE HERE
import sys

MOD = 998244353

def mod_pow(base, exp, mod):
    result = 1
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp //= 2
    return result

def mod_inv(x, mod):
    return mod_pow(x, mod - 2, mod)

def count_ways(n, m):
    if m == 1:
        return 0 if n > 1 else 1
    if n == 2:
        return m * (m - 1) % MOD
    total_ways = mod_pow(m - 1, n - 1, MOD)
    same_ways = mod_pow(m - 1, n - 2, MOD)
    result = (m * (total_ways - same_ways)) % MOD
    return result

input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])

print(count_ways(N, M))