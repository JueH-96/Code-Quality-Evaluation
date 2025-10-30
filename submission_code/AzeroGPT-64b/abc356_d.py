import math

def main():
    N, M = map(int, input().split())
    MOD = 998244353
    a = int(math.log2(M))
    c = M ^ (1 << a)
    d = N ^ (1 << a)
    e = N >> (a + 1)
    f = (1 << a) - 1
    s = popcount_in_range(0, c, d - 1) % MOD * fast_pow(2, a, MOD) % MOD
    t = sum([popcount_in_range(i * (1 << a), i * (1 << a) + c, min((i + 1) * (1 << a) - 1, d - 1)) for i in range(e + 2)])
    print((s + t) % MOD)

def popcount_in_range(s, x, y):
    # Return the total number of population count within the range [x, y]
    if x > y:
        return 0
    if s == 0:
        return x.bit_count() + popcount_in_range(s + 1, x >> 1, y >> 1)
    return x.bit_count() + popcount_in_range(s + 1, x >> 1, y >> 1) + popcount_in_range(s, x, y) - popcount_in_range(s, 0, (x ^ y) >> 1)

def fast_pow(base, exp, mod):
    result = 1
    while exp > 0:
        if exp & 1:
            result = result * base % mod
        base = base * base % mod
        exp >>= 1
    return result