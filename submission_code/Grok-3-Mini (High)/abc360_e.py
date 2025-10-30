import sys
MOD = 998244353

# Read input
data = sys.stdin.read().split()
N = int(data[0])
K = int(data[1])

# Compute coefficients modulo MOD
coeff1_mod = (N + 1) % MOD
coeff2_mod = (N - 1) % MOD
base_N_minus_2 = (N - 2) % MOD

# Compute powers modulo MOD
pow_N = pow(N, K, MOD)
pow_N_minus_2 = pow(base_N_minus_2, K, MOD)

# Compute P modulo MOD
P_mod = ((coeff1_mod * pow_N % MOD) - (coeff2_mod * pow_N_minus_2 % MOD) + MOD) % MOD

# Compute Q modulo MOD
Q_mod = (2 * pow_N % MOD) % MOD

# Compute modular inverse of Q using Fermat's Little Theorem
inv_Q = pow(Q_mod, MOD - 2, MOD)

# Compute the result
result = (P_mod * inv_Q % MOD) % MOD

# Output the result
print(result)