mod_val = 998244353

N = int(input())
d = len(str(N))
a = pow(10, d, mod_val)

if a == 1:
    S = N % mod_val
else:
    numerator = (pow(a, N, mod_val) - 1) % mod_val
    denominator = (a - 1) % mod_val
    inv_denominator = pow(denominator, mod_val - 2, mod_val)
    S = (numerator * inv_denominator) % mod_val

result = ( (N % mod_val) * S ) % mod_val
print(result)