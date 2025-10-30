import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    S_str = sys.stdin.readline().strip()
    MOD = 998244353

    # Precompute L_tuples from masks
    # mask_to_L_list[mask_val] stores the tuple (L_0, ..., L_N)
    # where L_j = LCS(S[0...j-1], T_prefix_yielding_mask)
    # L_0 = 0 by definition.
    # L_{j+1} - L_j is the j-th bit of mask_val (0-indexed bit).
    mask_to_L_list = [None] * (1 << N)
    for mask_val in range(1 << N):
        L = [0] * (N + 1)
        # L[0] is already 0
        for i in range(N): # bit i of mask_val determines L[i+1]-L[i]
            L[i+1] = L[i] + ((mask_val >> i) & 1)
        mask_to_L_list[mask_val] = tuple(L)

    # Precomputation for transitions:
    # transition_table[current_mask] will be a list of (next_mask, count_factor) tuples.
    # count_factor is how many characters 'a'...'z' lead from current_mask to next_mask.
    
    # memo_next_mask stores (L_tuple, char_code_0_to_25) -> next_mask
    # This helps avoid recomputing next_mask for same (L_tuple, char) pair.
    memo_next_mask = {}

    def get_next_mask_from_L(current_L_tuple, char_val_0_to_25):
        state_key = (current_L_tuple, char_val_0_to_25)
        if state_key in memo_next_mask:
            return memo_next_mask[state_key]
        
        char_to_append = chr(ord('a') + char_val_0_to_25)
        
        # next_L_list will store (L'_0, ..., L'_N) for T_prefix + char_to_append
        next_L_list = [0] * (N + 1) 
        # L'_0 is always 0
        
        for j in range(1, N + 1): # Compute L'_j for S[0...j-1]
            # S_str is 0-indexed, S_str[j-1] is the j-th character of S
            if S_str[j-1] == char_to_append:
                # Match S_str[j-1] with char_to_append.
                # LCS length increases by 1 based on LCS for S[0...j-2] with T_prefix.
                # current_L_tuple[j-1] is LCS(S[0...j-2], T_prefix)
                next_L_list[j] = current_L_tuple[j-1] + 1
            else:
                # S_str[j-1] != char_to_append.
                # Standard LCS recurrence: max of (don't use S_str[j-1]) and (don't use char_to_append).
                # LCS(S[0...j-2], T_prefix + char_to_append) is next_L_list[j-1].
                # LCS(S[0...j-1], T_prefix) is current_L_tuple[j].
                next_L_list[j] = max(current_L_tuple[j], next_L_list[j-1])
        
        # Convert next_L_list to its mask representation
        new_mask = 0
        for i in range(N): # bit i of mask is L'_{i+1} - L'_i
            if next_L_list[i+1] - next_L_list[i] == 1:
                new_mask |= (1 << i)
        
        memo_next_mask[state_key] = new_mask
        return new_mask

    transition_table = [[] for _ in range(1 << N)]
    
    S_unique_chars = sorted(list(set(S_str)))
    
    # Find a character value (0-25) that is not in S_str.
    # This is used for the "other characters" transition.
    char_val_not_in_S = -1 
    for k_val in range(26): # 'a' corresponds to 0, 'b' to 1, ...
        if chr(ord('a') + k_val) not in S_unique_chars:
            char_val_not_in_S = k_val
            break
            
    for current_mask_val in range(1 << N):
        current_L_tuple = mask_to_L_list[current_mask_val]
        
        # aggregated_transitions maps: next_mask -> number_of_chars_leading_to_it
        aggregated_transitions = {} 
        
        for char_s_unique in S_unique_chars:
            char_val = ord(char_s_unique) - ord('a')
            nm = get_next_mask_from_L(current_L_tuple, char_val)
            aggregated_transitions[nm] = aggregated_transitions.get(nm, 0) + 1
        
        num_other_chars = 26 - len(S_unique_chars)
        if num_other_chars > 0:
            # All num_other_chars behave identically. Use char_val_not_in_S as representative.
            # char_val_not_in_S must have been found if num_other_chars > 0.
            nm = get_next_mask_from_L(current_L_tuple, char_val_not_in_S)
            aggregated_transitions[nm] = aggregated_transitions.get(nm, 0) + num_other_chars
        
        for nm, factor in aggregated_transitions.items():
             transition_table[current_mask_val].append((nm, factor))

    # DP calculation
    # dp[len_T][mask] = number of strings T of length len_T with LCS profile 'mask'
    dp = [[0] * (1 << N) for _ in range(M + 1)]
    dp[0][0] = 1 # Empty string T, LCS profile is all zeros (mask 0)

    for length_T in range(M): # Current length of T prefixes
        for current_mask in range(1 << N):
            if dp[length_T][current_mask] == 0:
                continue
            
            current_count = dp[length_T][current_mask]
            
            # Iterate over precomputed aggregated transitions for this current_mask
            for next_mask, factor in transition_table[current_mask]:
                dp[length_T + 1][next_mask] = \
                    (dp[length_T + 1][next_mask] + current_count * factor) % MOD
    
    # Final answers
    # ans[k] = sum of dp[M][mask] where LCS length is k
    # LCS length for a mask (LCS(S, T)) is L_N.
    # L_N for mask_val is popcount(mask_val) due to our mask definition.
    ans = [0] * (N + 1)
    for mask_val in range(1 << N):
        if dp[M][mask_val] == 0:
            continue
        
        # LCS length is L_N from mask_to_L_list[mask_val][N]
        lcs_len_k = mask_to_L_list[mask_val][N]
        
        ans[lcs_len_k] = (ans[lcs_len_k] + dp[M][mask_val]) % MOD
        
    sys.stdout.write(" ".join(map(str, ans)) + "
")

solve()