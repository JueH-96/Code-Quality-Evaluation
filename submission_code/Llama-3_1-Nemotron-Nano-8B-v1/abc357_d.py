MOD = 998244353

N = int(input())
L = len(str(N))
r = pow(10, L, MOD)
N_mod = N % MOD

if r == 1:
    sum_geometric = N_mod
else:
    numerator = (pow(r, N, MOD) - 1) % MOD
    denominator = (r - 1) % MOD
    inv_denominator = pow(denominator, MOD - 2, MOD)
    sum_geometric = (numerator * inv_denominator) % MOD

result = (N_mod * sum_geometric) % MOD
print(result)