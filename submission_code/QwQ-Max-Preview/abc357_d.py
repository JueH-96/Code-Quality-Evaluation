mod = 998244353

N = int(input())
d = len(str(N))
D = (pow(10, d, mod) - 1) % mod
exponent = d * N
numerator = (pow(10, exponent, mod) - 1) % mod

if D != 0:
    inv_D = pow(D, mod - 2, mod)
    S = (numerator * inv_D) % mod
else:
    S = N % mod

result = ( (N % mod) * S ) % mod
print(result)