def pow_mod(base, exp, mod):
    result = 1
    base %= mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp //= 2
    return result

def mod_inverse(a, mod):
    return pow_mod(a, mod - 2, mod)

N, K = map(int, input().split())
MOD = 998244353

# Compute (N - 2)^K mod MOD
n_minus_2_pow_k = pow_mod(N - 2, K, MOD)

# Compute N^K mod MOD
n_pow_k = pow_mod(N, K, MOD)

# Compute P = (N+1) * N^K - (N - 1) * (N - 2)^K mod MOD
P = ((N + 1) * n_pow_k % MOD - (N - 1) * n_minus_2_pow_k % MOD + MOD) % MOD

# Compute Q = 2 * N^K mod MOD
Q = (2 * n_pow_k) % MOD

# Compute R = P * Q^(-1) mod MOD
Q_inv = mod_inverse(Q, MOD)
R = (P * Q_inv) % MOD

print(R)