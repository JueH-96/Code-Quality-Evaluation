M = 998244353
N = int(input())
str_N = str(N)
L = len(str_N)
A_pow_L = pow(10, L, M)
if A_pow_L == 1:
    result = (N * N) % M
else:
    exp = L * N
    pow_LN = pow(10, exp, M)
    num = (pow_LN - 1) % M
    den_mod = (A_pow_L - 1) % M
    inv_den = pow(den_mod, M - 2, M)
    sum_part = (num * inv_den) % M
    result = (N * sum_part) % M
print(result)