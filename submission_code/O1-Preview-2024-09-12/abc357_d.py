# YOUR CODE HERE
mod = 998244353
N = int(input())
S = str(N)
L = len(S)
x = N % mod
a = pow(10, L, mod)
denominator = (a - 1) % mod
if denominator == 0:
    S_sum = N % mod
else:
    numerator = (pow(a, N, mod) - 1 + mod) % mod
    inverse = pow(denominator, mod - 2, mod)
    S_sum = numerator * inverse % mod
V_N_mod = x * S_sum % mod
print(V_N_mod)