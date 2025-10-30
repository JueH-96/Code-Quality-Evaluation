def mod_inverse(a, mod):
    return pow(a, mod - 2, mod)

N, K = map(int, input().split())
MOD = 998244353

# E[X_K] = (N+1)/2 - (N-1)/2 * ((N-2)/N)^K

n_plus_1 = (N + 1) % MOD
n_minus_1 = (N - 1) % MOD
inv_2 = mod_inverse(2, MOD)

if N <= 2:
    power = 0
else:
    n_minus_2 = (N - 2) % MOD
    n_mod = N % MOD
    n_inv = mod_inverse(n_mod, MOD)
    base = (n_minus_2 * n_inv) % MOD
    power = pow(base, K, MOD)

term1 = (n_plus_1 * inv_2) % MOD
term2 = (n_minus_1 * inv_2 * power) % MOD

result = (term1 - term2 + MOD) % MOD
print(result)