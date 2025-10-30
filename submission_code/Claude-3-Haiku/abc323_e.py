from math import gcd

def modular_inverse(a, m):
    """
    Compute the modular inverse of a modulo m.
    """
    g, x, y = extended_gcd(a, m)
    if g != 1:
        raise ValueError('Modular inverse does not exist')
    return x % m

def extended_gcd(a, b):
    """
    Compute the extended Euclidean algorithm for a and b.
    Returns the gcd, and the coefficients x and y such that ax + by = gcd(a, b).
    """
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y

def solve():
    N, X = map(int, input().split())
    T = list(map(int, input().split()))

    total_time = sum(T)
    mod = 998244353

    numerator = 0
    for i in range(N):
        if (T[0] + i * sum(T)) % mod <= X:
            numerator += 1

    denominator = N * total_time
    g = gcd(numerator, denominator)
    numerator //= g
    denominator //= g

    inverse = modular_inverse(denominator, mod)
    result = (numerator * inverse) % mod
    print(result)

solve()