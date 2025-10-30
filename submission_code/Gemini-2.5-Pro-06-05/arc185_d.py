# YOUR CODE HERE
import sys

def solve():
    """
    Solves the problem by calculating the expected number of operations using a derived formula.
    """
    N, M = map(int, sys.stdin.readline().split())
    MOD = 998244353

    # The case N=1 corresponds to a simple path graph 0-1-2-...-M.
    # The expected number of moves to paint all vertices 1..M starting from 0 is M^2.
    if N == 1:
        ans = (M % MOD) * (M % MOD) % MOD
        print(ans)
        return

    # For the general case, a closed-form formula for the expected value can be derived.
    # The formula is: E = N*M*(M-1)/2 + N*M*(N+1)/2 + N*M*(N-1)*H_{N-1}
    # where H_{k} is the k-th harmonic number.

    # We need to compute H_{N-1} mod MOD. This requires modular inverses.
    # Precomputing inverses and harmonic numbers up to N-1.
    inv = [0] * (N + 1)
    H = [0] * (N + 1)
    
    if N > 1:
        inv[1] = 1
        H[1] = 1
        for i in range(2, N):
            # Modular inverse using the property: inv[i] = -(MOD/i) * inv[MOD % i]
            inv[i] = MOD - (MOD // i) * inv[MOD % i] % MOD
            H[i] = (H[i - 1] + inv[i]) % MOD

    # Modular inverse of 2
    inv2 = pow(2, MOD - 2, MOD)
    
    # Convert N and M to their values modulo MOD for calculations
    N_ = N % MOD
    M_ = M % MOD
    
    # Calculate each term of the formula modulo MOD.
    
    # Term 1: N * M * (M - 1) / 2
    term1 = M_ * (M_ - 1 + MOD) % MOD
    term1 = term1 * inv2 % MOD
    term1 = term1 * N_ % MOD
    
    # Term 2: N * M * (N + 1) / 2
    term2 = N_ * (N_ + 1) % MOD
    term2 = term2 * inv2 % MOD
    term2 = term2 * M_ % MOD
    
    # Term 3: N * M * (N - 1) * H_{N-1}
    term3 = 0
    if N > 1:
        H_N_minus_1 = H[N - 1]
        term3 = N_ * (N_ - 1 + MOD) % MOD
        term3 = term3 * M_ % MOD
        term3 = term3 * H_N_minus_1 % MOD
        
    # Sum the terms to get the final answer.
    ans = (term1 + term2 + term3) % MOD
    print(ans)

solve()