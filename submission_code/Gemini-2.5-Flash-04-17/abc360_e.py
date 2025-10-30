import sys

# Set recursion depth limit for potentially deep calls, though not expected here.
# sys.setrecursionlimit(2000)

MOD = 998244353

def power(base, exp, mod):
    """
    Calculates (base^exp) % mod using modular exponentiation.
    Handles base = 0, exp = 0 case according to typical modular exponentiation libraries (result 1).
    However, for exp > 0, if base is 0, the result is 0.
    Constraints: exp >= 1, so pow(0, K, mod) is 0 if base is 0.
    """
    return pow(base, exp, mod)

def mod_inverse(a, mod):
    """
    Calculates the modular multiplicative inverse of a modulo mod.
    Assumes mod is a prime number.
    Uses Fermat's Little Theorem: a^(mod-2) = a^(-1) mod mod for prime mod.
    """
    return power(a, mod - 2, mod)

def solve():
    """
    Calculates the expected position of the black ball after K operations modulo MOD.
    """
    N, K = map(int, sys.stdin.readline().split())

    # If N=1, there's only one ball (black) at position 1.
    # No swap can happen as a and b must be 1. Expected position is always 1.
    if N == 1:
        print(1)
        return

    # Formula for the expected position after K steps (1-based indexing):
    # E = (N+1)/2 - (N-1)/2 * (1 - 2/N)^K

    # Calculate required modular inverses
    half = mod_inverse(2, MOD) # 2^(-1) mod MOD
    N_inv = mod_inverse(N, MOD) # N^(-1) mod MOD

    # Calculate the base of the power term: (1 - 2/N) mod MOD
    # (1 - 2 * N_inv) mod MOD
    two_times_N_inv = (2 * N_inv) % MOD
    base = (1 - two_times_N_inv + MOD) % MOD # Ensure result is non-negative

    # Calculate the power term: (1 - 2/N)^K mod MOD
    term_power_of_c = power(base, K, MOD)

    # Calculate the two terms in the formula:
    # term1 = (N+1)/2 mod MOD = (N+1) * half mod MOD
    N_plus_1 = (N + 1) % MOD
    term1 = (N_plus_1 * half) % MOD

    # term2_coeff = (N-1)/2 mod MOD = (N-1) * half mod MOD
    N_minus_1 = (N - 1) % MOD
    term2_coeff = (N_minus_1 * half) % MOD

    # Calculate the second main term: term2_coeff * term_power_of_c mod MOD
    term2 = (term2_coeff * term_power_of_c) % MOD

    # Calculate the final result: (term1 - term2) mod MOD
    # Ensure result is non-negative
    result = (term1 - term2 + MOD) % MOD

    print(result)

# Read input and run solve
solve()