import sys

def solve():
    N = int(sys.stdin.readline())
    A_list_problem_indices = list(map(int, sys.stdin.readline().split())) # A_1 to A_{N-1}

    MOD = 998244353

    prime_factorizations = [] # For A_0 to A_{N-2} (0-indexed)
    distinct_primes_overall = set()

    for val_a in A_list_problem_indices:
        factors = {}
        d = 2
        temp_a = val_a
        while d * d <= temp_a:
            while temp_a % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp_a //= d
            d += 1
        if temp_a > 1:
            factors[temp_a] = factors.get(temp_a, 0) + 1
        
        prime_factorizations.append(factors)
        for p in factors:
            distinct_primes_overall.add(p)

    sorted_distinct_primes = sorted(list(distinct_primes_overall))
    prime_to_idx = {p: i for i, p in enumerate(sorted_distinct_primes)}
    
    # Using dict for sparse DP table
    # dp_curr[mask] stores sum of \prod P_j^{2(j+1)-N} terms
    dp_curr = {0: 1}

    # k_loop_idx iterates from 0 to N-2, representing A_0 to A_{N-2} (0-indexed A_list)
    # This corresponds to problem indices A_1 to A_{N-1}
    # The 'k' in P_k^{2k-N} is problem index j+1
    for k_loop_idx in range(N - 1):
        # Current A value being processed is A_list_problem_indices[k_loop_idx]
        # or prime_factorizations[k_loop_idx]
        current_A_factors = prime_factorizations[k_loop_idx]
        
        current_distinct_primes_in_A = sorted(list(current_A_factors.keys()))
        num_distinct_curr = len(current_distinct_primes_in_A)

        dp_next = {}
        
        # Problem index for P is k_loop_idx + 1
        term_exponent = 2 * (k_loop_idx + 1) - N

        for prev_mask, prev_sum_val in dp_curr.items():
            if prev_sum_val == 0:
                continue

            # Iterate over $2^{num_distinct_curr}$ choices for P_k, Q_k for current A
            for i in range(1 << num_distinct_curr):
                P_val = 1
                
                new_mask_contribution_from_Q = 0 
                valid_choice = True
                
                for bit_pos in range(num_distinct_curr):
                    p = current_distinct_primes_in_A[bit_pos]
                    # p_full_power is p^{v_p(A_k)}. This is an integer, not mod MOD.
                    p_full_power = p ** current_A_factors[p] 
                    p_global_idx = prime_to_idx[p]

                    if (i >> bit_pos) & 1: # p^{v_p(A_k)} goes to P_val
                        P_val = P_val * p_full_power # Can make P_val large, but P_val <= 1000
                        if (prev_mask >> p_global_idx) & 1: # p was in Q_j for j < k
                            valid_choice = False
                            break 
                    else: # p^{v_p(A_k)} goes to Q_val
                        new_mask_contribution_from_Q |= (1 << p_global_idx)
                
                if not valid_choice:
                    continue

                new_mask = prev_mask | new_mask_contribution_from_Q
                
                # pow handles negative exponent correctly in Python 3.8+
                term_for_Pval = pow(P_val, term_exponent, MOD)
                
                current_dp_val = dp_next.get(new_mask, 0)
                dp_next[new_mask] = (current_dp_val + prev_sum_val * term_for_Pval) % MOD
        
        dp_curr = dp_next

    sum_P_terms = 0
    for val in dp_curr.values():
        sum_P_terms = (sum_P_terms + val) % MOD

    constant_factor_A = 1
    # k_loop_idx from 0 to N-2 for A_list_problem_indices
    # problem index k is k_loop_idx + 1
    for k_loop_idx in range(N - 1):
        A_val = A_list_problem_indices[k_loop_idx]
        exponent_A = N - (k_loop_idx + 1) # N-k
        term_for_Ak = pow(A_val, exponent_A, MOD)
        constant_factor_A = (constant_factor_A * term_for_Ak) % MOD
        
    final_ans = (sum_P_terms * constant_factor_A) % MOD
    
    print(final_ans)

solve()