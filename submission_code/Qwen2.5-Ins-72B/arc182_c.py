# YOUR CODE HERE
import sys
from math import gcd
from collections import Counter

MOD = 998244353

def mod_pow(x, n, mod):
    res = 1
    while n > 0:
        if n & 1:
            res = res * x % mod
        x = x * x % mod
        n >>= 1
    return res

def mod_inv(x, mod):
    return mod_pow(x, mod - 2, mod)

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

def count_divisors(factors):
    counter = Counter(factors)
    divisors = 1
    for count in counter.values():
        divisors *= (count + 1)
    return divisors

def solve(N, M):
    primes = prime_factors(M)
    unique_primes = set(primes)
    prime_powers = {p: 0 for p in unique_primes}
    
    for p in unique_primes:
        power = 0
        while M % p == 0:
            M //= p
            power += 1
        prime_powers[p] = power
    
    total_score = 0
    for k in range(1, N + 1):
        for i in range(1, M + 1):
            factors = prime_factors(i)
            divisors = count_divisors(factors)
            total_score += divisors * mod_pow(M, k, MOD)
            total_score %= MOD
    
    return total_score

N, M = map(int, sys.stdin.readline().strip().split())
print(solve(N, M))