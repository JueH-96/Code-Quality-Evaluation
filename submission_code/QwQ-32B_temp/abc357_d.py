N = int(input().strip())
m = 998244353
d = len(str(N))
r = pow(10, d, m)

if r == 1:
    sum_mod = N % m
else:
    pow_r_N = pow(r, N, m)
    numerator = (pow_r_N - 1) % m
    denominator = (r - 1) % m
    inv_denominator = pow(denominator, m-2, m)
    sum_mod = (numerator * inv_denominator) % m

n_mod = N % m
result = (n_mod * sum_mod) % m
print(result)