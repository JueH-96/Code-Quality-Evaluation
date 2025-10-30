MOD = 998244353
s = input().strip()
L = len(s)
num_mod = int(s) % MOD
N = int(s)
base = pow(10, L, MOD)

if base == 1:
    S = N % MOD
else:
    numerator = (pow(base, N, MOD) - 1) % MOD
    denominator = (base - 1) % MOD
    inv_denominator = pow(denominator, MOD - 2, MOD)
    S = (numerator * inv_denominator) % MOD

result = (num_mod * S) % MOD
print(result)