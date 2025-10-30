import sys

def solve():
    N_str = sys.stdin.readline().strip()
    N = int(N_str) # N can be up to 10^18

    MOD = 998244353

    # L is the number of digits in N
    L = len(N_str)

    # V_N = N_value * sum_{i=0}^{N_repetitions-1} ( (10^L)^i )
    # In this problem, N_value is the integer N.
    # And N_repetitions is also N.

    # (N_value % MOD)
    # This is the integer N, modulo MOD.
    n_val_mod = N % MOD

    # The geometric series part: sum_{i=0}^{N-1} X^i where X = 10^L
    # The base of the geometric series is X = (10^L) % MOD
    x_geom_base = pow(10, L, MOD)
    
    # The number of terms in the geometric series is N (N_repetitions).
    # This N is also the exponent in the geometric sum formula part X^N.
    num_geom_terms = N 

    geom_sum_mod_val = 0
    if x_geom_base == 1:
        # If X = 1, sum = 1 + 1 + ... + 1 (num_geom_terms times) = num_geom_terms
        geom_sum_mod_val = num_geom_terms % MOD
    else:
        # If X != 1, sum = (X^num_geom_terms - 1) / (X - 1)
        
        # Numerator part: (X^num_geom_terms - 1) % MOD
        # pow(base, exp, mod) computes (base^exp) % mod efficiently.
        pow_x_n = pow(x_geom_base, num_geom_terms, MOD)
        
        # (pow_x_n - 1) could be -1 if pow_x_n is 0. Add MOD to ensure positive before taking modulo.
        numerator = (pow_x_n - 1 + MOD) % MOD

        # Denominator part: (X - 1) % MOD
        # (x_geom_base - 1) could be -1 if x_geom_base is 0 (not possible for 10^L if MOD doesn't divide 10).
        # Add MOD to ensure positive.
        denominator = (x_geom_base - 1 + MOD) % MOD
        
        # Modular inverse of denominator: (X - 1)^(MOD-2) % MOD
        # This is valid because MOD is prime and x_geom_base != 1, so denominator != 0.
        inv_denominator = pow(denominator, MOD - 2, MOD)

        geom_sum_mod_val = (numerator * inv_denominator) % MOD
        
    # Final result: (N_value % MOD * geom_sum % MOD) % MOD
    result = (n_val_mod * geom_sum_mod_val) % MOD
    
    sys.stdout.write(str(result) + "
")

if __name__ == '__main__':
    solve()