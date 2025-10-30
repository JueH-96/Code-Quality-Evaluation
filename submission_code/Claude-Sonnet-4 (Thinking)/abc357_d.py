def mod_pow(base, exp, mod):
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result

def mod_inverse(a, mod):
    return mod_pow(a, mod - 2, mod)

N = int(input())
MOD = 998244353

# Number of digits in N
d = len(str(N))

# Compute 10^d mod MOD
ten_pow_d = mod_pow(10, d, MOD)

# Compute 10^(N*d) mod MOD
# By Fermat's little theorem, we reduce the exponent modulo (MOD - 1)
exp = (N * d) % (MOD - 1)
ten_pow_nd = mod_pow(10, exp, MOD)

# Compute (10^(N*d) - 1) mod MOD
numerator = (ten_pow_nd - 1 + MOD) % MOD

# Compute (10^d - 1) mod MOD
denominator = (ten_pow_d - 1 + MOD) % MOD

# Compute modular inverse of denominator
inv_denominator = mod_inverse(denominator, MOD)

# Compute V_N mod MOD
result = (N % MOD) * numerator % MOD * inv_denominator % MOD

print(result)