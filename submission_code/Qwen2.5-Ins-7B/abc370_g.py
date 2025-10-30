# YOUR CODE HERE
from math import isqrt
from functools import cache

MOD = 998244353

@cache
def divisors_sum(n):
    if n == 1:
        return 1
    divisors = 1
    for p in range(2, isqrt(n) + 1):
        if n % p == 0:
            exp = 0
            while n % p == 0:
                n //= p
                exp += 1
            divisors *= (p**(exp + 1) - 1) // (p - 1)
    if n > 1:
        divisors *= (n + 1)
    return divisors

def good_integer(n):
    return divisors_sum(n) % 3 == 0

@cache
def count_good_sequences(n, m):
    if n == 0 or m == 0:
        return 1
    if n < 0:
        return 0
    if m == 1:
        return int(good_integer(n))
    return (count_good_sequences(n - 1, m - 1) + count_good_sequences(n, m - 1)) % MOD

n, m = map(int, input().split())
print(count_good_sequences(n, m))