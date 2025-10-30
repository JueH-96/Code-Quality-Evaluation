MOD = 998244353

N = int(input())
D = len(str(N))
a = pow(10, D, MOD)

if a == 1:
    ans = (N % MOD) * (N % MOD) % MOD
else:
    denominator = (a - 1) % MOD
    inv_denominator = pow(denominator, MOD - 2, MOD)
    exponent = D * N
    pow_10_DN = pow(10, exponent, MOD)
    numerator = (pow_10_DN - 1) % MOD
    temp = (numerator * inv_denominator) % MOD
    ans = ((N % MOD) * temp) % MOD

print(ans)