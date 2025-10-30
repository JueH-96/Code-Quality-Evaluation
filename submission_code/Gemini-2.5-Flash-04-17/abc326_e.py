# Required functions for modular arithmetic
def power(base, exp, mod):
    """Computes (base^exp) % mod using binary exponentiation."""
    res = 1
    base %= mod
    while exp > 0:
        if exp % 2 == 1:
            res = (res * base) % mod
        base = (base * base) % mod
        exp //= 2
    return res

def modInverse(n, mod):
    """Computes modular multiplicative inverse of n modulo mod."""
    # Fermat's Little Theorem applies since mod is prime and n is not a multiple of mod.
    # Problem constraints guarantee N >= 1 and N <= 3e5, while MOD is 998244353.
    # So N is not a multiple of MOD.
    return power(n, mod - 2, mod)

# Read input
N = int(input())
A = list(map(int, input().split()))

# Modulus
MOD = 998244353

# Calculate invN = N^(-1) mod MOD
# N >= 1, so N is not a multiple of MOD
invN = modInverse(N, MOD)

# Calculate R = (N+1)/N mod MOD = (N+1) * N^(-1) mod MOD
# N+1 will not be a multiple of MOD for N <= 3e5 < MOD - 1
# (N + 1) % MOD is just N + 1 since N + 1 < MOD
R = ((N + 1) * invN) % MOD

# The expected value E_0 is given by the formula:
# E_0 = (1/N) * sum_{j=1}^N ((N+1)/N)^(j-1) * A_j mod MOD
# Using 0-indexed array A[0], ..., A[N-1] for A_1, ..., A_N
# Let R = (N+1)/N mod MOD.
# E_0 = (1/N) * sum_{k=0}^{N-1} R^k * A[k] mod MOD
# E_0 = invN * (sum_{k=0}^{N-1} R^k * A[k]) mod MOD

# Calculate the sum S = sum_{k=0}^{N-1} R^k * A[k] mod MOD
total_sum_S = 0
current_power_of_R = 1 # R^0 = 1

for k in range(N):
    # Add term A[k] * R^k to the sum
    # A[k] is already in [0, MOD-1] as per constraints 0 <= A_i < MOD
    term = (A[k] * current_power_of_R) % MOD
    total_sum_S = (total_sum_S + term) % MOD

    # Update current_power_of_R to R^(k+1) for the next iteration
    current_power_of_R = (current_power_of_R * R) % MOD

# The final expected value E_0 = invN * S mod MOD
ans = (invN * total_sum_S) % MOD

# Print the result
print(ans)