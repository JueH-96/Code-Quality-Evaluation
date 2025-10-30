MOD = 998244353

def solve():
    N, M = map(int, input().split())
    B_in = list(map(int, input().split())) # 0-indexed B_in

    # all_dp_true[k][x] will store dp[k][true][x]
    # all_dp_false[k][x] will store dp[k][false][x]
    # k is the current index, from 0 to N-1.
    # dp table is for suffix A_k ... A_{N-1}
    # Base cases are for k=N (empty suffix)
    all_dp_true = [[0]*(M + 1) for _ in range(N + 1)]
    all_dp_false = [[0]*(M + 1) for _ in range(N + 1)]

    # Base cases for dp[N][state][X] (empty suffix A_N ... A_{N-1})
    # If X is met (true state), 0 ways for empty suffix.
    # If X is not met (false state), 1 way for empty suffix.
    for x_val in range(M + 1): # Includes x_val=0, though problem values are 1..M
        all_dp_false[N][x_val] = 1
        all_dp_true[N][x_val] = 0
    
    # Fill DP table: k_idx from N-1 down to 0
    for k_idx in range(N - 1, -1, -1):
        # cur_dp_true = all_dp_true[k_idx]
        # cur_dp_false = all_dp_false[k_idx]
        
        # Values for k_idx+1 (suffix A_{k_idx+1} ... A_{N-1})
        # dp_k_plus_1_true_arr = all_dp_true[k_idx+1]
        # dp_k_plus_1_false_arr = all_dp_false[k_idx+1]

        for x_val in range(1, M + 1): # S_i = X must be >= 1
            dp_k_plus_1_true_at_x = all_dp_true[k_idx+1][x_val]
            dp_k_plus_1_false_at_x = all_dp_false[k_idx+1][x_val]

            if B_in[k_idx] != -1:
                val_B_k = B_in[k_idx]
                if val_B_k > x_val: # A_k > X, not in component relative to X limit
                    all_dp_true[k_idx][x_val] = dp_k_plus_1_true_at_x
                    all_dp_false[k_idx][x_val] = dp_k_plus_1_false_at_x
                elif val_B_k == x_val: # A_k = X, in component, X is met
                    all_dp_true[k_idx][x_val] = (dp_k_plus_1_true_at_x + dp_k_plus_1_false_at_x) % MOD
                    all_dp_false[k_idx][x_val] = 0 
                else: # val_B_k < x_val: A_k < X, in component, X not met by this A_k
                    all_dp_true[k_idx][x_val] = dp_k_plus_1_true_at_x
                    all_dp_false[k_idx][x_val] = dp_k_plus_1_false_at_x
            else: # B_in[k_idx] == -1 (A_k is variable)
                # Case 1: A_k > X. (M-X) choices. Not in component.
                # (M - x_val) can be negative if MOD not added, but M >= x_val.
                choices_gt_x = (M - x_val + MOD) % MOD 
                term1_true = (choices_gt_x * dp_k_plus_1_true_at_x) % MOD
                term1_false = (choices_gt_x * dp_k_plus_1_false_at_x) % MOD
                
                # Case 2: A_k = X. 1 choice. In component. X is met.
                term2_true = (dp_k_plus_1_true_at_x + dp_k_plus_1_false_at_x) % MOD
                # term2_false = 0, A_k=X means X is met.
                
                # Case 3: A_k < X. (X-1) choices. In component. X not met by this A_k.
                choices_lt_x = (x_val - 1 + MOD) % MOD # if x_val=1, choices_lt_x=0
                
                term3_true = (choices_lt_x * dp_k_plus_1_true_at_x) % MOD
                term3_false = (choices_lt_x * dp_k_plus_1_false_at_x) % MOD
                
                all_dp_true[k_idx][x_val] = (term1_true + term2_true + term3_true) % MOD
                all_dp_false[k_idx][x_val] = (term1_false + term3_false) % MOD
    
    # N_star_val[x] stores N_*(i-1, x) when considering vertex i.
    # Initially, for i=0 (vertex 1), prefix is empty. N_*(-1, x) = 1.
    N_star_val = [1] * (M + 1) 
    
    total_f_sum = 0
    
    for i_idx in range(N): # Iterate through vertex i (0-indexed)
        # Current N_star_val is N_*(i_idx-1, X)
        
        current_i_contribution = 0
        for x_val in range(1, M + 1): # S_i = X, must be >= 1
            term_N_star = N_star_val[x_val]
            # Ways for suffix A_{i_idx}...A_{N-1} to have max X in CC of i_idx is all_dp_true[i_idx][x_val]
            term_dp_i_true = all_dp_true[i_idx][x_val] 
            current_i_contribution = (current_i_contribution + term_N_star * term_dp_i_true) % MOD
            
        total_f_sum = (total_f_sum + current_i_contribution) % MOD
        
        # Update N_star_val for next iteration i_idx+1.
        # It should become N_*(i_idx, X). Multiply by factor for B_in[i_idx] (current A_{i_idx}).
        if i_idx < N - 1: # No need to update after last i_idx
            for x_s_max in range(M + 1): # S_max from previous elements (now including A_{i_idx})
                                         # Condition for A_{i_idx} is A_{i_idx} > x_s_max
                factor_for_B_i = 0
                if B_in[i_idx] != -1:
                    if B_in[i_idx] > x_s_max: # Fixed A_{i_idx} satisfies condition
                        factor_for_B_i = 1
                    # else: factor_for_B_i = 0 (fixed A_{i_idx} does not satisfy)
                else: # B_in[i_idx] == -1 (A_{i_idx} is variable)
                    # A_{i_idx} must be > x_s_max. Choices are x_s_max+1, ..., M.
                    # Number of choices is M - x_s_max.
                    # (M - x_s_max) can be negative if MOD not added. M >= x_s_max.
                    factor_for_B_i = (M - x_s_max + MOD) % MOD
                
                N_star_val[x_s_max] = (N_star_val[x_s_max] * factor_for_B_i) % MOD
            
    print(total_f_sum)

solve()