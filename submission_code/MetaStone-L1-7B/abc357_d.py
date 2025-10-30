MOD = 998244353

N = int(input())
M = len(str(N))
r = pow(10, M, MOD)
exponent = N

numerator = (pow(r, exponent, MOD) - 1) % MOD

if r == 1:
    S = exponent % MOD
else:
    denominator = (r - 1) % MOD
    inv_denominator = pow(denominator, MOD - 2, MOD)
    S = (numerator * inv_denominator) % MOD

V = (N % MOD) * S % MOD
print(V)