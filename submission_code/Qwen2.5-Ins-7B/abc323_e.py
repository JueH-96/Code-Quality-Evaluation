# YOUR CODE HERE
from math import gcd
from functools import reduce

def modinv(a, mod):
    return pow(a, mod-2, mod)

def lcm(a, b):
    return a * b // gcd(a, b)

n, x = map(int, input().split())
t = list(map(int, input().split()))
total = sum(t)
mod = 998244353

# Calculate the probability
numerator = 0
for i in range(n):
    if (x + 0.5) % t[i] < 0.5:
        numerator += t[i] - ((x + 0.5) % t[i])
    else:
        numerator += (x + 0.5) % t[i]
    numerator %= total

# Calculate the denominator
denominator = reduce(lcm, t) * n

# Simplify the fraction
common_divisor = gcd(numerator, denominator)
numerator //= common_divisor
denominator //= common_divisor

# Calculate the modular inverse of the denominator
denominator_inv = modinv(denominator, mod)

# Calculate the result
result = (numerator * denominator_inv) % mod

print(result)