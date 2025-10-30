import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    MOD = 998244353

    if M == 1:
        # If M=1, all a_i must be 1. Product is 1. tau(1)=1.
        # There are N sequences: (1), (1,1), ..., (1,...,1) (N times). Each has score 1.
        # Sum of scores is N.
        print(N % MOD)
        return

    # Determine primes <= M
    primes = []
    # Explicitly list primes for M <= 16 to avoid complex prime generation logic
    # Order matters for mapping to indices consistently.
    potential_primes = [2, 3, 5, 7, 11, 13]
    for p_cand in potential_primes:
        if p_cand <= M:
            primes.append(p_cand)
    R = len(primes)

    # Precompute exponents e_p(val)
    # exp_val[val][prime_idx] = exponent of primes[prime_idx] in val
    exp_val = [[0] * R for _ in range(M + 1)]
    for val in range(1, M + 1):
        num = val
        for i, p in enumerate(primes):
            if p > num : break # Optimization
            while num > 0 and num % p == 0:
                exp_val[val][i] += 1
                num //= p
    
    # Precompute C(U) = sum_{val=1 to M} product_{p in U} e_p(val)
    # U is represented by a bitmask for primes in P
    C_val = [0] * (1 << R) # C_val[mask] stores C(mask)
    for val in range(1, M + 1):
        for mask in range(1 << R):
            prod_ep_val = 1
            for i in range(R):
                if (mask >> i) & 1: # if prime p_i is in U (mask)
                    prod_ep_val *= exp_val[val][i]
            C_val[mask] = (C_val[mask] + prod_ep_val) # No modulo here yet, sum then modulo
    for mask in range(1 << R):
        C_val[mask] %= MOD
            
    # dp_A[mask][k_val] = sum of product C(Q_i) for partitions of set 'mask' into 'k_val' blocks
    dp_A = [[0] * (R + 1) for _ in range(1 << R)]
    dp_A[0][0] = 1 # Empty set has one partition into 0 blocks, product is 1.

    for k_val in range(1, R + 1): # k_val is number of blocks in partition
        for mask_val in range(1 << R): # mask_val is the set to be partitioned
            if mask_val == 0 : # Empty set cannot be partitioned into k_val >= 1 blocks
                dp_A[mask_val][k_val] = 0
                continue

            # Find smallest index prime in mask_val (p_first_idx)
            # Q1 (first block in partition) must contain this prime to avoid overcounting.
            p_first_idx = 0
            while not ((mask_val >> p_first_idx) & 1):
                p_first_idx += 1
            
            mask_val_without_p_first = mask_val ^ (1 << p_first_idx)
            
            # Iterate sub_loop_mask over subsets of mask_val_without_p_first
            # Q1_mask = sub_loop_mask | (1 << p_first_idx)
            current_sub_iter = 0
            while True: # Iterate current_sub_iter over all submasks of mask_val_without_p_first
                q1_mask = current_sub_iter | (1 << p_first_idx)
                
                remaining_mask_for_dp = mask_val ^ q1_mask # mask_val \ q1_mask
                
                if dp_A[remaining_mask_for_dp][k_val-1] != 0:
                    term = dp_A[remaining_mask_for_dp][k_val-1] * C_val[q1_mask]
                    dp_A[mask_val][k_val] = (dp_A[mask_val][k_val] + term) % MOD
                
                if current_sub_iter == mask_val_without_p_first:
                    break
                current_sub_iter = (current_sub_iter + 1) | mask_val_without_p_first # This finds next submask
                current_sub_iter &= mask_val_without_p_first                     # by finding next number that is submask


    # Compute S_m(K) = sum_{j=0 to K} (j+m)!/j! * M^j
    # S_comp[m_val][k_idx] stores S_{m_val}(N-k_idx)
    # K goes from N down to N-R. So k_idx from 0 to R.
    S_comp = [[0] * (R + 1) for _ in range(R + 1)]
    inv_M_minus_1 = pow(M - 1, MOD - 2, MOD)
    
    M_powers_cache = {} # Cache M^X % MOD

    for k_idx in range(R + 1): 
        K = N - k_idx
        if K < 0: # Sum S_m(K) is 0 if K < 0. All S_comp[...][k_idx] remain 0.
            continue

        # Base case S_0(K)
        key_K_plus_1 = K + 1
        if key_K_plus_1 not in M_powers_cache:
             M_powers_cache[key_K_plus_1] = pow(M, key_K_plus_1, MOD)
        M_pow_K_plus_1 = M_powers_cache[key_K_plus_1]
        
        s0_k = (M_pow_K_plus_1 - 1 + MOD) % MOD * inv_M_minus_1 % MOD
        S_comp[0][k_idx] = s0_k
        
        # Recursive step for S_m(K)
        for m_val in range(1, R + 1):
            # (K+m_val)! / K! * M^(K+1)
            # factor (K+m_val)...(K+1)
            term1_factor_K_m = 1
            for i in range(m_val):
                term1_factor_K_m = term1_factor_K_m * ((K + m_val - i) % MOD) % MOD
            
            # M_pow_K_plus_1 is already computed for S_0(K)
            term1 = term1_factor_K_m * M_pow_K_plus_1 % MOD
            
            term2_m_S_prev = m_val * S_comp[m_val-1][k_idx] % MOD
            
            S_comp[m_val][k_idx] = (term1 - term2_m_S_prev + MOD) % MOD * inv_M_minus_1 % MOD

    # Total Score Calculation
    if N == 0: # No sequences of length >= 1
        total_score = 0
    else:
        # Term for Q = empty set: sum_{k=1 to N} M^k = S_0(N) - M^0 = S_comp[0][0] - 1
        total_score = (S_comp[0][0] - 1 + MOD) % MOD
    
    for mask_val in range(1, 1 << R): # Iterate Q (non-empty)
        s_num_primes_in_Q = 0 # popcount(mask_val)
        for i in range(R):
            if (mask_val >> i) & 1:
                s_num_primes_in_Q += 1
        
        f_Q_term = 0
        for m_blocks in range(1, s_num_primes_in_Q + 1):
            # Need S_m(N-m). This is S_comp[m_blocks][m_blocks] (k_idx = m_blocks for K=N-m_blocks)
            # If N-m_blocks < 0, then K < 0. S_comp[m_blocks][m_blocks] would be 0 due to K<0 check.
            s_m_Nm = S_comp[m_blocks][m_blocks]

            term = dp_A[mask_val][m_blocks] * s_m_Nm % MOD
            f_Q_term = (f_Q_term + term) % MOD
        
        total_score = (total_score + f_Q_term) % MOD
        
    print(total_score)

solve()