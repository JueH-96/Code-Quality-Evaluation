# YOUR CODE HERE
import sys
from math import gcd

MOD = 998244353

def modinv(a, m):
    g, x, y = extended_gcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m

def extended_gcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = extended_gcd(b % a, a)
        return (g, x - (b // a) * y, y)

def solve(n, a):
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = a[i - 1]
        for j in range(1, i):
            dp[i] += dp[j] * (i - j) * pow(i, MOD - 2, MOD)
            dp[i] %= MOD
        dp[i] += a[i - 1] * (i - 1) * pow(i, MOD - 2, MOD)
        dp[i] %= MOD
    result = 0
    for i in range(1, n + 1):
        result += dp[i] * pow(n, MOD - 2, MOD)
        result %= MOD
    return result

n = int(input())
a = list(map(int, input().split()))
print(solve(n, a))