MOD = 998244353

n_str = input().strip()
d = len(n_str)
N = int(n_str)

r = pow(10, d, MOD)
denominator = (r - 1) % MOD
inv_denominator = pow(denominator, MOD - 2, MOD)

exponent = d * N
pow_10 = pow(10, exponent, MOD)
numerator = (pow_10 - 1) % MOD

sum_s = (numerator * inv_denominator) % MOD
result = (N % MOD) * sum_s % MOD

print(result)