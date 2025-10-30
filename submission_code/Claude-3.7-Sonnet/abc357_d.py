# YOUR CODE HERE
def compute_remainder(n, mod):
    # Get the number of digits in n
    n_str = str(n)
    L = len(n_str)
    
    # Compute the remainder using the formula: V_N = N * (10^(L*N) - 1) / (10^L - 1)
    # For modular arithmetic, we need to compute modular inverse of (10^L - 1)
    
    # First, compute 10^(L*N) % mod using Fermat's Little Theorem
    # Since mod is prime, 10^(mod-1) ≡ 1 (mod mod)
    phi = mod - 1
    reduced_exponent = (L * n) % phi
    
    # Calculate (10^(L*N) - 1) % mod
    numerator = (pow(10, reduced_exponent, mod) - 1) % mod
    
    # Calculate (10^L - 1) % mod
    denominator = (pow(10, L, mod) - 1) % mod
    
    # Calculate modular inverse of denominator using Fermat's Little Theorem
    # For prime modulus p, a^(p-2) ≡ a^(-1) (mod p)
    inv_denominator = pow(denominator, mod - 2, mod)
    
    # Calculate the final result: N * (10^(L*N) - 1) / (10^L - 1) % mod
    result = (n % mod * numerator % mod * inv_denominator % mod) % mod
    
    return result

# Get input
n = int(input().strip())
mod = 998244353

# Compute and print the result
print(compute_remainder(n, mod))