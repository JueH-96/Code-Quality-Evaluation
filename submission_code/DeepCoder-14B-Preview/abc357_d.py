MOD = 998244353

n = int(input())
k = len(str(n))
r = pow(10, k, MOD)

if r == 1:
    s = n % MOD
else:
    inv_r_minus_1 = pow(r - 1, MOD - 2, MOD)
    pow_r_n = pow(r, n, MOD)
    numerator = (pow_r_n - 1) % MOD
    s = (numerator * inv_r_minus_1) % MOD

v_n = (n % MOD) * s % MOD
print(v_n)