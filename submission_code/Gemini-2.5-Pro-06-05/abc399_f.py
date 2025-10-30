# YOUR CODE HERE
import sys

def solve():
    """
    Reads input, solves the problem, and prints the answer.
    """
    MOD = 998244353

    def combinations_init(max_val, mod):
        """
        Precomputes factorials and their modular inverses to create a combination function.
        """
        fact = [1] * (max_val + 1)
        inv_fact = [1] * (max_val + 1)
        for i in range(1, max_val + 1):
            fact[i] = (fact[i - 1] * i) % mod
        
        inv_fact[max_val] = pow(fact[max_val], mod - 2, mod)
        for i in range(max_val - 1, -1, -1):
            inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % mod
        
        def combinations(n, k):
            if k < 0 or k > n:
                return 0
            return (fact[n] * inv_fact[k] * inv_fact[n - k]) % mod
        
        return combinations

    # Use fast I/O
    input = sys.stdin.readline

    # Read problem inputs
    try:
        N, K = map(int, input().split())
        A = list(map(int, input().split()))
    except (IOError, ValueError):
        # Handle cases with no input, e.g., in some testing environments
        return

    # Calculate prefix sums of the sequence A.
    # P[i] = A_0 + ... + A_{i-1}
    # P has size N+1, with P[0] = 0.
    P = [0] * (N + 1)
    for i in range(N):
        P[i + 1] = (P[i] + A[i]) % MOD

    # Precompute combinations C(K, k) for efficiency
    comb = combinations_init(K, MOD)
    
    total_sum = 0
    
    # Q[m] will store the sum of m-th powers of prefix sums seen so far.
    # At the beginning of loop for j, Q[m] = sum_{i=0}^{j-1} P[i]^m
    Q = [0] * (K + 1)
    
    # Initialize Q with powers of P[0].
    # Since P[0] = 0, P[0]^0 = 1, and P[0]^m = 0 for m > 0.
    Q[0] = 1
    
    # Iterate through j from 1 to N.
    for j in range(1, N + 1):
        # Calculate powers of P[j] from 0 to K.
        Pj_powers = [0] * (K + 1)
        Pj_powers[0] = 1
        for i in range(1, K + 1):
            Pj_powers[i] = (Pj_powers[i - 1] * P[j]) % MOD
            
        current_j_contribution = 0
        for k in range(K + 1):
            m = K - k
            
            # term = C(K,k) * P[j]^k * Q[m]
            term = (Pj_powers[k] * Q[m]) % MOD
            term = (comb(K, k) * term) % MOD
            
            # Apply the sign (-1)^m
            if m % 2 == 1:
                # Odd power of -1 results in subtraction
                current_j_contribution = (current_j_contribution - term + MOD) % MOD
            else:
                # Even power of -1 results in addition
                current_j_contribution = (current_j_contribution + term) % MOD

        total_sum = (total_sum + current_j_contribution) % MOD
        
        # Update Q for the next iteration (j+1) by adding the powers of P[j].
        for m in range(K + 1):
            Q[m] = (Q[m] + Pj_powers[m]) % MOD
            
    print(total_sum)

solve()