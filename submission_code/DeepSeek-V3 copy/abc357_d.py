# YOUR CODE HERE
MOD = 998244353

def compute_v_n(N):
    # Calculate the length of N when represented as a string
    length = len(str(N))
    # Calculate 10^(length * N)
    power = pow(10, length * N, MOD)
    # Calculate (10^(length * N) - 1) / (10^length - 1)
    denominator = pow(10, length, MOD) - 1
    if denominator == 0:
        return (N * N) % MOD
    # Since MOD is prime, we can compute the modular inverse of denominator
    # Using Fermat's little theorem: a^(MOD-2) ≡ a^(-1) mod MOD
    inv_denominator = pow(denominator, MOD - 2, MOD)
    numerator = (power - 1) % MOD
    # Compute V_N mod MOD
    v_n = (N * numerator * inv_denominator) % MOD
    return v_n

# Read input
N = int(input())
# Compute and print the result
print(compute_v_n(N))