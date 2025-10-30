import sys

def main():
    N, K = map(int, sys.stdin.readline().split())
    A_input = list(map(int, sys.stdin.readline().split()))

    MOD = 998244353

    # 1. Compute prefix sums P
    # P[i] stores sum of A_input[0] ... A_input[i-1] (0-indexed A_input)
    # P[0] = 0
    # P[k] in math (sum of first k elements) corresponds to P[k] in code.
    # P has N+1 elements, P[0] through P[N].
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = (P[i] + A_input[i]) % MOD

    # 2. Precompute powers pow_P[i][s] = P[i]^s % MOD
    # Dimensions: (N+1) for index i (from 0 to N for P_i), 
    #             (K+1) for exponent s (from 0 to K)
    pow_P = [[0] * (K + 1) for _ in range(N + 1)]

    for i in range(N + 1):
        pow_P[i][0] = 1 # P_i^0 = 1 (even for P_i=0, as 0^0=1 in this context)
        current_P_val = P[i] 
        for s in range(1, K + 1):
            pow_P[i][s] = (pow_P[i][s-1] * current_P_val) % MOD
            # If P_i=0, P_i^1 = 1*0=0, P_i^2=0*0=0, etc. This is correct.
    
    # 3. Precompute binomial coefficients C_binom[m] = K choose m
    C_binom = [0] * (K + 1)
    C_binom[0] = 1
    # Using C(n,k) = C(n,k-1) * (n-k+1) / k
    for m_val in range(1, K + 1):
        term = C_binom[m_val-1]
        term = (term * (K - m_val + 1)) % MOD
        term = (term * pow(m_val, MOD - 2, MOD)) % MOD # Modular inverse for division by m_val
        C_binom[m_val] = term
    
    total_ans = 0
    
    # Iterate m from the binomial expansion (sum over m from 0 to K)
    # m_binomial_idx is 'm' in P_r^m and C(K,m)
    for m_binomial_idx in range(K + 1):
        # Determine sign: (-1)^(K-m_binomial_idx)
        sign = 1
        if (K - m_binomial_idx) % 2 == 1:
            sign = MOD - 1 # This is -1 % MOD
        
        # Coefficient for this m_binomial_idx: C(K,m_binomial_idx) * (-1)^(K-m_binomial_idx)
        coeff_m = (C_binom[m_binomial_idx] * sign) % MOD
        
        # current_sum_Pj_pow_Km will store Sum_{j=0}^{r-1} P_j^{K-m_binomial_idx} as r iterates
        current_sum_Pj_pow_Km = 0 
        
        # term_for_m_sum will store Sum_{r=1 to N} P_r^{m_binomial_idx} * (Sum_{j=0 to r-1} P_j^{K-m_binomial_idx})
        # This is the content of the square brackets in the formula for the current m_binomial_idx
        term_for_m_sum = 0    
        
        # Loop r_idx from 1 to N (for P_r)
        # P[r_idx] means P_r in math notation (sum of first r_idx elements from A_input)
        for r_idx in range(1, N + 1):
            # Add P[r_idx-1]^(K-m_binomial_idx) to current_sum_Pj_pow_Km
            # P[r_idx-1] is P_{r-1} in math. This is the P_j term for j=r-1.
            # The exponent is K-m_binomial_idx.
            val_P_j_pow_K_minus_m = pow_P[r_idx-1][K - m_binomial_idx]
            current_sum_Pj_pow_Km = (current_sum_Pj_pow_Km + val_P_j_pow_K_minus_m) % MOD
            
            # Fetch P[r_idx]^{m_binomial_idx}
            val_P_r_pow_m = pow_P[r_idx][m_binomial_idx]
            
            # Add to term_for_m_sum: P[r_idx]^{m_binomial_idx} * current_sum_Pj_pow_Km
            term_for_m_sum = (term_for_m_sum + val_P_r_pow_m * current_sum_Pj_pow_Km) % MOD
            
        # Add to total_ans: coeff_m * term_for_m_sum
        total_ans = (total_ans + coeff_m * term_for_m_sum) % MOD
        
    sys.stdout.write(str(total_ans) + "
")

if __name__ == '__main__':
    main()