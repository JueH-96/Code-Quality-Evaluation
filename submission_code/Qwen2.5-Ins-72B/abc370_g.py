# YOUR CODE HERE
import sys
from math import gcd
from collections import Counter

MOD = 998244353

def is_good(n):
    total = sum(i for i in range(1, n + 1) if n % i == 0)
    return total % 3 == 0

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

def mod_pow(x, y, p):
    res = 1
    x = x % p
    while y > 0:
        if y & 1:
            res = (res * x) % p
        y = y >> 1
        x = (x * x) % p
    return res

def mod_inv(x, p):
    return mod_pow(x, p - 2, p)

def solve(N, M):
    good_integers = [i for i in range(1, N + 1) if is_good(i)]
    good_integers_set = set(good_integers)
    
    dp = [0] * (N + 1)
    dp[1] = 1
    
    for i in range(1, N + 1):
        if i in good_integers_set:
            dp[i] = 1
            for j in range(2, M + 1):
                dp[i] = (dp[i] * i) % MOD
                if dp[i] > N:
                    break
            if dp[i] <= N:
                dp[i] = 1
            else:
                dp[i] = 0
    
    for i in range(2, N + 1):
        for j in range(i, N + 1, i):
            if j in good_integers_set:
                dp[j] = (dp[j] + dp[i]) % MOD
    
    result = 0
    for i in good_integers:
        result = (result + dp[i]) % MOD
    
    return result

N, M = map(int, input().split())
print(solve(N, M))