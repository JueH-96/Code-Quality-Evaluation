# YOUR CODE HERE
import sys

# Define the modulus
MOD = 998244353

# Modular exponentiation function
def fast_pow(base, power):
    """
    Compute (base^power) % MOD using binary exponentiation.
    """
    # The pow built-in function with three arguments is efficient for modular exponentiation
    return pow(base, power, MOD)

# Modular inverse function using Fermat's Little Theorem
def inverse(a):
    """
    Compute modular multiplicative inverse of a modulo MOD.
    Assumes MOD is prime and a % MOD != 0.
    """
    # Ensure a is in the range [0, MOD-1]
    a %= MOD
    
    # Check for edge case a=0. Based on problem constraints N >= 1 and N <= MOD-1, N % MOD will never be 0.
    # Also, 2 % MOD is never 0 since MOD is an odd prime.
    # if a == 0:
    #     raise ValueError("Modular inverse does not exist for 0.") # This path should not be reached

    # Use Fermat's Little Theorem: a^(MOD-2) % MOD is the inverse of a mod MOD
    # This relies on MOD being a prime number. 998244353 is prime.
    return pow(a, MOD - 2, MOD)

# Main logic function
def solve():
    # Read N and K from standard input
    N, K = map(int, sys.stdin.readline().split())

    # The expected position E_k satisfies the recurrence relation:
    # E_{k+1} = E_k * (N-2)/N + (N+1)/N
    # This is a linear recurrence relation E_{k+1} = c * E_k + d
    # with c = (N-2)/N and d = (N+1)/N.
    # The initial condition is E_0 = 1 (black ball starts at position 1).
    # The closed-form solution is E_K = c^K * (E_0 - E*) + E*
    # where E* is the fixed point E* = d / (1-c).
    # E* = ((N+1)/N) / (1 - (N-2)/N) = ((N+1)/N) / ( (N - (N-2))/N )
    # E* = ((N+1)/N) / (2/N) = (N+1)/2.
    # E_0 = 1. So E_0 - E* = 1 - (N+1)/2 = (2 - (N+1))/2 = (1-N)/2.
    # Thus, the formula is E_K = ((N-2)/N)^K * (1-N)/2 + (N+1)/2.
    # All calculations need to be performed modulo MOD.

    # Calculate components modulo MOD
    # N % MOD: Use Python's % operator which handles large N correctly.
    N_mod = N % MOD
    
    # Need modular inverses of N and 2 modulo MOD
    # Constraints: 1 <= N <= MOD-1 implies N_mod is in [1, MOD-1]. Thus N_mod != 0.
    invN = inverse(N_mod) 
    # MOD=998244353 is an odd prime, so 2 % MOD != 0.
    inv2 = inverse(2) # This equals (MOD+1)//2

    # Calculate c = (N-2)/N mod MOD
    # We need (N-2) mod MOD. If N=1, N-2=-1. (N_mod - 2 + MOD) % MOD ensures a positive result [0, MOD-1].
    num_c = (N_mod - 2 + MOD) % MOD
    c = (num_c * invN) % MOD

    # Calculate c^K mod MOD using modular exponentiation
    c_pow_K = fast_pow(c, K)

    # Calculate D_0 = E_0 - E* = (1-N)/2 mod MOD
    # We need (1-N) mod MOD. (1 - N_mod + MOD) % MOD ensures positive result.
    num_D0 = (1 - N_mod + MOD) % MOD
    D0_mod = (num_D0 * inv2) % MOD

    # Calculate E* = (N+1)/2 mod MOD
    # N_mod + 1 is always positive since N >= 1.
    num_Estar = (N_mod + 1) % MOD 
    Estar_mod = (num_Estar * inv2) % MOD

    # Combine using the formula: E_K = (c^K * D_0 + E*) mod MOD
    # Perform calculations using modular arithmetic properties
    term1 = (c_pow_K * D0_mod) % MOD
    E_K = (term1 + Estar_mod) % MOD
    
    # Print the final result to standard output
    print(E_K)

# This check ensures the solve() function runs when the script is executed directly
if __name__ == '__main__':
    solve()