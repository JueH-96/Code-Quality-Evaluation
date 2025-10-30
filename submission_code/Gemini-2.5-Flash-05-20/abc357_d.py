import sys

def solve():
    # Read N as a string first to easily get its length (number of digits)
    N_str = sys.stdin.readline().strip()
    N = int(N_str)
    
    MOD = 998244353

    # L is the number of digits in N
    L = len(N_str) 

    # Calculate P = 10^L % MOD. This will be the common ratio in the geometric series.
    P = pow(10, L, MOD)

    # V_N = N * (1 + (10^L) + (10^L)^2 + ... + (10^L)^(N-1))
    # This is N * (sum of a geometric series)
    # Let r = 10^L. The sum is 1 + r + r^2 + ... + r^(N-1)
    
    if P == 1:
        # Special case: If r (which is P) is 1, the sum of N terms is N.
        # So, V_N = N * N.
        # We need (N * N) % MOD. Since N can be very large, take N % MOD.
        result = (N % MOD * N % MOD) % MOD
    else:
        # General case: If r (which is P) is not 1, the sum is (r^N - 1) / (r - 1).
        # So, V_N = N * (P^N - 1) / (P - 1).
        
        # Calculate (P^N - 1) % MOD.
        # pow(P, N, MOD) computes P^N % MOD. N is the exponent here.
        # Add MOD before modulo to handle cases where (pow(P, N, MOD) - 1) might be negative.
        numerator_sum_term = (pow(P, N, MOD) - 1 + MOD) % MOD
        
        # Calculate the modular inverse of (P - 1) using Fermat's Little Theorem: a^(MOD-2) % MOD.
        # (P - 1) is guaranteed not to be 0 here because the P == 1 case is already handled.
        denominator_inverse = pow(P - 1, MOD - 2, MOD)

        # N in the formula N * (sum) needs to be taken modulo MOD for multiplication.
        N_mod = N % MOD

        # Combine all terms modulo MOD.
        # Result = (N % MOD * (P^N - 1) % MOD * (P - 1)^(-1) % MOD) % MOD
        result = (N_mod * numerator_sum_term % MOD * denominator_inverse % MOD) % MOD

    sys.stdout.write(str(result) + '
')

# Call the solve function to run the program
solve()