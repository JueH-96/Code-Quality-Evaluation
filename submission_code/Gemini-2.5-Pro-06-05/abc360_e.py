import sys

def solve():
    """
    Reads input, solves the problem, and prints the result.
    """
    try:
        line = sys.stdin.readline()
        if not line:
            return
        N, K = map(int, line.split())
    except (IOError, ValueError):
        return

    MOD = 998244353

    def modInverse(n, modulus):
        """
        Calculates the modular inverse of n modulo modulus using Fermat's Little Theorem.
        This is efficient as MOD is a prime number.
        """
        return pow(n, modulus - 2, modulus)

    # Let E_k be the expected position of the black ball after k operations.
    # The derived closed-form solution for E_K is:
    # E_K = E_f + (E_0 - E_f) * A^K
    # where:
    # E_0 = 1 (initial position)
    # E_f = (N+1)/2 (long-term equilibrium expected position)
    # A = 1 - 2/N = (N-2)/N

    # Handle the edge case where N=1. The black ball is always at position 1.
    if N == 1:
        print(1)
        return

    # All calculations must be done modulo MOD.

    # Calculate inv2 = 2^(-1) mod MOD and invN = N^(-1) mod MOD
    inv2 = modInverse(2, MOD)
    invN = modInverse(N, MOD)

    # Calculate E_f = (N+1)/2 mod MOD
    term_N_plus_1 = (N + 1) % MOD
    E_f = (term_N_plus_1 * inv2) % MOD

    # Calculate (E_0 - E_f) mod MOD. E_0 = 1.
    E0_minus_Ef = (1 - E_f + MOD) % MOD

    # Calculate A = (N-2)/N mod MOD
    A_num = (N - 2 + MOD) % MOD
    A = (A_num * invN) % MOD

    # Calculate A^K mod MOD using modular exponentiation
    A_pow_K = pow(A, K, MOD)

    # Calculate the second term: (E_0 - E_f) * A^K mod MOD
    second_term = (E0_minus_Ef * A_pow_K) % MOD

    # Calculate final E_K = E_f + second_term mod MOD
    E_K = (E_f + second_term) % MOD

    print(E_K)

solve()