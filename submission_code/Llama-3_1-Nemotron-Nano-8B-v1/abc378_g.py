A, B, M = map(int, input().split())

def comb(n, k, mod):
    if n < 0 or k < 0 or k > n:
        return 0
    numerator = 1
    for i in range(n, n - k, -1):
        numerator = numerator * i % mod
    denominator = 1
    for i in range(1, k + 1):
        denominator = denominator * i % mod
    return numerator * pow(denominator, mod - 2, mod) % mod

def catalan(n, mod):
    if n == 0:
        return 1
    c = [0] * (n + 1)
    c[0] = 1
    for i in range(1, n + 1):
        c[i] = (c[i - 1] * (2 * i) * comb(2 * i, i, mod) // (i + 1)) % mod
    return c[n]

# The solution involves the product of two Catalan numbers
cat_a = catalan(A, M)
cat_b = catalan(B, M)
print((cat_a * cat_b) % M)