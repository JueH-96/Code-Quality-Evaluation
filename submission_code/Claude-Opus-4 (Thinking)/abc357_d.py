n = int(input())
mod = 998244353

# Find number of digits in n
d = len(str(n))

# Compute 10^d mod mod
ten_d = pow(10, d, mod)

# Compute 10^(n*d) mod mod
ten_nd = pow(ten_d, n, mod)

# Compute (10^(n*d) - 1) mod mod
numerator = (ten_nd - 1 + mod) % mod

# Compute (10^d - 1) mod mod
denominator = (ten_d - 1 + mod) % mod

# Compute modular inverse of denominator
inv_denominator = pow(denominator, mod - 2, mod)

# Compute v_n mod mod
v_n = (n % mod * numerator % mod * inv_denominator) % mod

print(v_n)