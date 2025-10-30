import sys

MOD = 998244353

def power(base, exp):
    """Computes (base^exp) % MOD."""
    res = 1
    base %= MOD
    while exp > 0:
        if exp % 2 == 1:
            res = (res * base) % MOD
        base = (base * base) % MOD
        exp //= 2
    return res

def modInverse(n):
    """Computes modular multiplicative inverse of n modulo MOD."""
    return power(n, MOD - 2)

def solve():
    N, K = map(int, sys.stdin.readline().split())

    # Special case for N=1: black ball is always at position 1.
    # The formula also correctly handles this:
    # E_K = (1+1)/2 - (1-1)/2 * ((1-2)/1)^K = 1 - 0 * (-1)^K = 1
    if N == 1:
        print(1)
        return

    # E_K = (N+1)/2 - ((N-2)/N)^K * (N-1)/2
    
    inv2 = modInverse(2)
    invN = modInverse(N)

    # Calculate the first term: (N+1)/2
    # Ensure numerator is within [0, MOD-1]
    term1_numerator = (N + 1) % MOD
    term1 = (term1_numerator * inv2) % MOD

    # Calculate the base for the power term: (N-2)/N
    # (N-2) can be negative if N=1, so add MOD before taking modulo.
    # For N=1, (1-2+MOD)%MOD = MOD-1.
    # For N=2, (2-2+MOD)%MOD = 0.
    # For N>2, N-2 is positive.
    term_pow_base_numerator = (N - 2 + MOD) % MOD 
    term_pow_base = (term_pow_base_numerator * invN) % MOD
    
    # Calculate ((N-2)/N)^K
    # The power function handles base=0 correctly (0^0=1, 0^K=0 for K>0)
    term_pow = power(term_pow_base, K)

    # Calculate (N-1)/2
    # N-1 is always non-negative for N >= 1
    term_Nminus1_inv2_numerator = (N - 1) % MOD
    term_Nminus1_inv2 = (term_Nminus1_inv2_numerator * inv2) % MOD

    # Calculate the second term: ((N-2)/N)^K * (N-1)/2
    term2 = (term_pow * term_Nminus1_inv2) % MOD

    # Final result: (term1 - term2) % MOD
    # Add MOD before modulo to handle potential negative result from subtraction
    ans = (term1 - term2 + MOD) % MOD
    
    print(ans)

solve()