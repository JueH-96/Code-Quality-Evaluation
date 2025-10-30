import sys

def solve():
    N, M, K = map(int, sys.stdin.readline().split())
    X_orig = list(map(int, sys.stdin.readline().split()))
    X = [val - 1 for val in X_orig] # 0-indexed characters

    MOD = 998244353

    is_all_same = True
    first_char = -1
    if M > 0:
        first_char = X[0]
        for i in range(1, M):
            if X[i] != first_char:
                is_all_same = False
                break
    
    is_ulc = False # Unique Last Character: X_M not in X_1...X_{M-1}
    if M > 0:
        is_ulc = True 
        last_char_val = X[M-1]
        for i in range(M - 1): # Iterate X_0 to X_{M-2}
            if X[i] == last_char_val:
                is_ulc = False
                break
    
    # Precompute f[L][R]: num sequences of length L using R specified chars, all of them.
    # f[L][R] = sum_{j=0 to R} (-1)^j * C(R,j) * (R-j)^L
    # Max L is N, max R is K-1.
    
    C = [[0] * (K + 1) for _ in range(K + 1)]
    for i in range(K + 1):
        C[i][0] = 1
        for j in range(1, i + 1):
            C[i][j] = (C[i-1][j-1] + C[i-1][j]) % MOD

    pow_val = [[0] * (N + 1) for _ in range(K + 1)]
    for base_val in range(K + 1): # base_val can be 0
        pow_val[base_val][0] = 1
        if base_val == 0: # 0^exp = 0 for exp > 0
            for exp_val in range(1, N + 1):
                pow_val[base_val][exp_val] = 0
        else:
            for exp_val in range(1, N + 1):
                pow_val[base_val][exp_val] = (pow_val[base_val][exp_val-1] * base_val) % MOD
    
    f_LR = [[0] * K for _ in range(N + 1)] # Max R is K-1
    for r_chars in range(K): 
        if r_chars == 0:
            f_LR[0][0] = 1
            continue
        for l_len in range(1, N + 1): # Min length to have r_chars is r_chars
            if l_len < r_chars:
                continue
            term_sum = 0
            for j_chosen_not_to_use in range(r_chars + 1):
                term = C[r_chars][j_chosen_not_to_use] * pow_val[r_chars - j_chosen_not_to_use][l_len]
                term %= MOD
                if j_chosen_not_to_use % 2 == 1:
                    term_sum = (term_sum - term + MOD) % MOD
                else:
                    term_sum = (term_sum + term) % MOD
            f_LR[l_len][r_chars] = term_sum

    if is_all_same:
        if M == 0: # Not possible by constraints M>=2
             print(0)
             return
        if K == 1: 
            print(0)
            return

        num_chars_in_segment = K - 1
        min_len_segment = num_chars_in_segment if num_chars_in_segment > 0 else 0
        
        target_sum_lengths = N - (M - 1)
        if target_sum_lengths < 0: 
            print(0)
            return
        if M * min_len_segment > target_sum_lengths: 
            print(0)
            return
        
        P_coeffs = [0] * (target_sum_lengths + 1)
        for l_val in range(min_len_segment, target_sum_lengths + 1):
            P_coeffs[l_val] = f_LR[l_val][num_chars_in_segment]

        dp_pow = [[0] * (target_sum_lengths + 1) for _ in range(M + 1)]
        dp_pow[0][0] = 1 

        for p_val in range(1, M + 1):
            for s_val in range(target_sum_lengths + 1): # Current sum for p_val-1 segments
                if dp_pow[p_val-1][s_val] == 0: 
                    continue
                for l_coeff_idx in range(min_len_segment, target_sum_lengths - s_val + 1): # Length of current segment
                    if P_coeffs[l_coeff_idx] == 0: 
                        continue
                    term = dp_pow[p_val-1][s_val] * P_coeffs[l_coeff_idx]
                    term %= MOD
                    dp_pow[p_val][s_val + l_coeff_idx] = (dp_pow[p_val][s_val + l_coeff_idx] + term) % MOD
        
        print(dp_pow[M][target_sum_lengths])

    elif is_ulc:
        if M < 2 : # Problem constraint M >= 2. P is X[0...M-2] is non-empty if M>=2.
             print(0) # If M=1, P is empty. This logic simplifies.
             return   # With M>=2, M-1 >= 1, so P always has at least one char.

        # dp[len][match_P_len][seen_X_M_flag]
        # P = X[0...M-2]. Length of P is M-1. match_P_len goes from 0 to M-1.
        dp = [[[0,0] for _ in range(M)] for _ in range(N + 1)] # M states for match_P_len: 0 to M-1
        dp[0][0][0] = 1 

        char_X_M = X[M-1]
        
        for length in range(N):
            for p_match_len in range(M): # p_match_len for P=X[0...M-2]
                                         # If p_match_len = M-1, P is fully matched.
                
                # Calculate for dp[length][p_match_len][0] (no X_M seen)
                if dp[length][p_match_len][0] > 0:
                    val_dp0 = dp[length][p_match_len][0]
                    # Option 1: Append char_X_M
                    # P-match does not change (X_M not in P). Flag becomes 1.
                    dp[length+1][p_match_len][1] = (dp[length+1][p_match_len][1] + val_dp0) % MOD
                    
                    # Option 2 & 3: Append char c != char_X_M
                    if p_match_len == M-1 : # P is already matched
                        # Any char c != char_X_M keeps P matched at M-1. (K-1 choices)
                        dp[length+1][M-1][0] = (dp[length+1][M-1][0] + val_dp0 * (K-1)) % MOD
                    else: # P not fully matched
                        char_to_match_P = X[p_match_len] # This is P[p_match_len]
                        # Option 2.1: Append char_to_match_P (must be != char_X_M as X_M not in P)
                        dp[length+1][p_match_len+1][0] = (dp[length+1][p_match_len+1][0] + val_dp0) % MOD
                        # Option 2.2: Append other char from S_K \ {char_X_M, char_to_match_P} (K-2 choices)
                        if K - 2 >= 0:
                            dp[length+1][p_match_len][0] = (dp[length+1][p_match_len][0] + val_dp0 * (K-2)) % MOD
                
                # Calculate for dp[length][p_match_len][1] (X_M seen)
                if dp[length][p_match_len][1] > 0:
                    val_dp1 = dp[length][p_match_len][1]
                    # Option 1: Append char_X_M
                    # P-match does not change. Flag stays 1.
                    dp[length+1][p_match_len][1] = (dp[length+1][p_match_len][1] + val_dp1) % MOD

                    # Option 2 & 3: Append char c != char_X_M
                    if p_match_len == M-1 : # P is already matched
                        dp[length+1][M-1][1] = (dp[length+1][M-1][1] + val_dp1 * (K-1)) % MOD
                    else: # P not fully matched
                        char_to_match_P = X[p_match_len]
                        # Option 2.1: Append char_to_match_P
                        dp[length+1][p_match_len+1][1] = (dp[length+1][p_match_len+1][1] + val_dp1) % MOD
                        # Option 2.2: Append other char (K-2 choices)
                        if K - 2 >= 0:
                            dp[length+1][p_match_len][1] = (dp[length+1][p_match_len][1] + val_dp1 * (K-2)) % MOD
        
        total_ans = 0
        min_len_suffix = K - 1 if K > 1 else 0
        num_chars_suffix = K - 1 if K > 1 else 0

        for k_amain_len in range(1, N + 1):
            # P is X[0...M-2]. Fully matched state for P is M-1.
            count_amain = dp[k_amain_len][M-1][1] # Must have seen X_M
            if count_amain == 0:
                continue

            len_suffix = N - k_amain_len
            if len_suffix < min_len_suffix: # Suffix too short
                continue
            
            count_suffix = f_LR[len_suffix][num_chars_suffix]
            if count_suffix == 0 and len_suffix > 0 : # len_suffix = 0, num_chars_suffix = 0 has f_LR[0][0]=1
                 if not (len_suffix == 0 and num_chars_suffix == 0): # check to ensure f_LR[0][0] is not wrongly zeroed
                    continue


            total_ans = (total_ans + count_amain * count_suffix) % MOD
            
        print(total_ans)

    else: # Not all_same and not ULC
        print(0)

solve()