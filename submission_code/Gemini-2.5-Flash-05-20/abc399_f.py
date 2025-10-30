import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    MOD = 998244353

    # Precompute factorials and inverse factorials for combinations C(n, k)
    # Since n will always be K (max 10), we only need factorials up to K.
    fact = [1] * (K + 1)
    inv_fact = [1] * (K + 1)
    for i in range(1, K + 1):
        fact[i] = (fact[i-1] * i) % MOD
    
    # Calculate modular inverse for fact[K] using Fermat's Little Theorem: a^(MOD-2) % MOD
    inv_fact[K] = pow(fact[K], MOD - 2, MOD)
    # Calculate inverse factorials for i from K-1 down to 0: inv_fact[i] = inv_fact[i+1] * (i+1)
    for i in range(K - 1, -1, -1):
        inv_fact[i] = (inv_fact[i+1] * (i+1)) % MOD

    # Helper function to compute nCr % MOD
    def nCr_mod_p(n, r):
        if r < 0 or r > n:
            return 0
        # For this problem, n is always K, and r is m.
        # Both K, m, and K-m are within the precomputed range.
        return (((fact[n] * inv_fact[r]) % MOD) * inv_fact[n-r]) % MOD

    # Precompute term coefficients for the binomial expansion: C(K, m) * (-1)^(K-m)
    term_coeffs = [0] * (K + 1)
    for m in range(K + 1):
        coeff = nCr_mod_p(K, m)
        if (K - m) % 2 == 1: # If K-m is odd, (-1)^(K-m) is -1 (or MOD-1)
            term_coeffs[m] = (MOD - coeff) % MOD
        else: # If K-m is even, (-1)^(K-m) is 1
            term_coeffs[m] = coeff
    
    # current_S stores the current prefix sum P_r (sum of A_1 to A_r)
    # current_prefix_sum_powers[p] stores sum_{j=0}^{r-1} (P_j)^p
    # P_0 = 0. By convention, 0^0 = 1. So (P_0)^0 = 1. (P_0)^p = 0 for p > 0.
    current_S = 0
    current_prefix_sum_powers = [0] * (K + 1)
    current_prefix_sum_powers[0] = 1 # Initialize with P_0^0 = 1. Other P_0^p are 0.

    total_overall_sum = 0

    # Iterate through each element of A (A[i] is A_{i+1} in 1-indexed problem statement)
    # This loop processes r from 1 to N.
    for i in range(N): 
        # Update current_S to be P_r (where r = i+1)
        current_S = (current_S + A[i]) % MOD
        
        # Calculate powers of the current prefix sum: (P_r)^p for p from 0 to K
        P_r_powers = [0] * (K + 1)
        P_r_powers[0] = 1 # P_r^0 is 1 (even if P_r itself is 0)
        for p in range(1, K + 1):
            P_r_powers[p] = (P_r_powers[p-1] * current_S) % MOD

        # Add contribution for the current 'r' to the total sum.
        # The term for a fixed r is: sum_{m=0}^K ( C(K, m) * (-1)^(K-m) * (P_r)^m * (sum_{j=0}^{r-1} (P_j)^{K-m}) )
        for m in range(K + 1):
            term_coeff = term_coeffs[m] # C(K, m) * (-1)^(K-m)
            
            P_r_m = P_r_powers[m] # (P_r)^m
            
            # The sum sum_{j=0}^{r-1} (P_j)^{K-m} is stored in current_prefix_sum_powers[K-m]
            # at this point (before P_r is included in current_prefix_sum_powers).
            sum_P_j_Km = current_prefix_sum_powers[K-m]
            
            term_val = (P_r_m * sum_P_j_Km) % MOD
            total_overall_sum = (total_overall_sum + term_coeff * term_val) % MOD

        # After using P_r for calculations, update current_prefix_sum_powers
        # by adding (P_r)^p to prepare for the next iteration (r+1).
        # Now current_prefix_sum_powers[p] will store sum_{j=0}^{r} (P_j)^p.
        for p in range(K + 1):
            current_prefix_sum_powers[p] = (current_prefix_sum_powers[p] + P_r_powers[p]) % MOD
            
    # Print the final result
    sys.stdout.write(str(total_overall_sum) + "
")

# Call the solver function
solve()