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

# Calculate number of digits in N
d = len(str(N))

# Calculate 10^d mod MOD
ten_d = mod_pow(10, d, MOD)

# Calculate 10^(N*d) mod MOD
# Since N*d can be very large, we need to be careful
# We can use the fact that 10^(N*d) = (10^d)^N
ten_nd = mod_pow(ten_d, N % (MOD - 1), MOD)

# Calculate numerator: N * (10^(N*d) - 1)
numerator = (N % MOD) * ((ten_nd - 1 + MOD) % MOD) % MOD

# Calculate denominator: 10^d - 1
denominator = (ten_d - 1 + MOD) % MOD

# Calculate result using modular inverse
result = (numerator * mod_inverse(denominator, MOD)) % MOD

print(result)