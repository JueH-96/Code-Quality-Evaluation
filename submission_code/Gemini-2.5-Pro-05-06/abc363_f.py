import sys

# It's good practice to increase recursion limit for deep DFS, though default might be enough.
# sys.setrecursionlimit(2000) # Default is often 1000. Max depth here is ~25.

def solve():
    N = int(sys.stdin.readline())

    # Global variable to store the first found solution
    ans_S = None

    # Helper: check if a string consists of digits '1'-'9'
    def is_valid_num_str(s):
        if not s: # Empty string is not a valid number string
            return False
        if s[0] == '0': # Leading zeros not allowed (str(k) won't have them unless k=0)
             return False # str(0) is "0", caught by '0' in s.
        if '0' in s:
            return False
        return True

    # Helper: check if a string is a palindrome
    def is_palindrome(s):
        return s == s[::-1]

    # Helper: reverse a string
    def reverse_str(s):
        return s[::-1]

    # Initial check: N itself as a palindromic number string (Form 1: S = Q)
    N_str = str(N)
    if is_valid_num_str(N_str) and is_palindrome(N_str):
        # Length check for S=Q is implicitly handled by overall length check for any S.
        # No, explicit check:
        if 1 <= len(N_str) <= 1000:
            print(N_str)
            return
    
    # DFS parameters
    MAX_DEPTH = 25  # Max number of P_i terms on one side
    K_0_LIMIT = 400000 # Iteration limit for val(P_i) candidates

    # DFS function
    # p_list: list of strings [P_1, P_2, ..., P_k]
    # n_rem: remaining product to achieve from inner part
    def dfs(n_rem, p_list):
        nonlocal ans_S 
        if ans_S is not None: # A solution has already been found
            return

        # Calculate sum of lengths of P_i strings chosen so far (for one side)
        current_sum_len_P_i_on_one_side = 0
        for p_s in p_list:
            current_sum_len_P_i_on_one_side += len(p_s)

        # --- Try to terminate with Q (Form 2 like ending) ---
        q_cand_str = str(n_rem)
        if is_valid_num_str(q_cand_str) and is_palindrome(q_cand_str):
            # Construct the full string S = P_1*...*P_k * Q * rev(P_k)*...*rev(P_1)
            # and check its length.
            
            # Parts of S: P_1, ..., P_k, Q, rev(P_k), ..., rev(P_1)
            s_parts_for_S = []
            current_total_len = 0

            for p_s_outer in p_list:
                s_parts_for_S.append(p_s_outer)
                current_total_len += len(p_s_outer)
            
            s_parts_for_S.append(q_cand_str)
            current_total_len += len(q_cand_str)
            
            for i in range(len(p_list) - 1, -1, -1): # Iterate P_k down to P_1
                p_s_to_rev = p_list[i]
                rev_ps_outer = reverse_str(p_s_to_rev)
                s_parts_for_S.append(rev_ps_outer)
                current_total_len += len(rev_ps_outer)
            
            num_stars = len(s_parts_for_S) - 1
            if num_stars < 0: num_stars = 0 # Handles S=Q case where p_list is empty
            
            current_total_len += num_stars

            if 1 <= current_total_len <= 1000:
                ans_S = "*".join(s_parts_for_S)
                return
        
        if ans_S is not None: return # Check if solution found

        # --- Try to terminate if n_rem is 1 (Form 3 like ending) ---
        if n_rem == 1:
            if not p_list: # S cannot be empty or just stars. Must have P_i terms.
                return
            
            # Construct S = P_1*...*P_k * rev(P_k)*...*rev(P_1)
            s_parts_for_S = []
            current_total_len = 0

            for p_s_outer in p_list:
                s_parts_for_S.append(p_s_outer)
                current_total_len += len(p_s_outer)
            
            # No Q part. Add reversed P_i strings.
            for i in range(len(p_list) - 1, -1, -1):
                p_s_to_rev = p_list[i]
                rev_ps_outer = reverse_str(p_s_to_rev)
                s_parts_for_S.append(rev_ps_outer)
                current_total_len += len(rev_ps_outer)
            
            num_stars = len(s_parts_for_S) - 1
            # num_stars will be >= 0 because p_list is not empty (e.g. P1*rev(P1) has 1 star)
            current_total_len += num_stars

            if 1 <= current_total_len <= 1000:
                ans_S = "*".join(s_parts_for_S)
                return
        
        if ans_S is not None: return # Check if solution found

        # --- Pruning based on depth and estimated length ---
        if len(p_list) >= MAX_DEPTH:
            return

        # Estimate minimum length if we add one more P_{k+1} (length at least 1) and a Q (length at least 1)
        # Min length for P_{k+1} is 1 (e.g., "1"). Min length for rev(P_{k+1}) is 1.
        # Current P_i strings contribute len on one side: current_sum_len_P_i_on_one_side
        # Add P_{k+1} (len 1) -> current_sum_len_P_i_on_one_side + 1
        # Total from P parts (both sides): 2 * (current_sum_len_P_i_on_one_side + 1)
        # Min Q len: 1 (e.g., "1")
        # Number of P terms on one side becomes len(p_list)+1.
        # Stars for Form 2 (with Q): 2 * (len(p_list)+1)
        min_len_estimate = 2 * (current_sum_len_P_i_on_one_side + 1) + 1 + (2 * (len(p_list) + 1))
        if min_len_estimate > 1000:
            return

        # --- Recursive step: try to add P_{k+1} ---
        # Iterate val_p_cand (value of P_{k+1})
        for val_p_cand in range(1, K_0_LIMIT + 1):
            s_p_cand = str(val_p_cand)

            if not is_valid_num_str(s_p_cand):
                continue
            
            # Pruning: if adding this s_p_cand already makes the sum of P_i lengths too large
            # Min Q ("1") + stars (2*(len(p_list)+1)) + P parts (2*(sum_len+len(s_p_cand)))
            len_Pi_one_side_with_new_cand = current_sum_len_P_i_on_one_side + len(s_p_cand)
            num_P_pairs_with_new_cand = len(p_list) + 1
            
            estimate_len_with_Q = 2 * len_Pi_one_side_with_new_cand + 1 + (2 * num_P_pairs_with_new_cand)
            if estimate_len_with_Q > 1000 :
                 # This check is important. If current s_p_cand is too long,
                 # and given len(str(k)) is non-decreasing for k > 0 (not strictly, e.g. "9" vs "11"),
                 # this specific candidate might be too long.
                 # It won't break the loop, as a shorter number string (e.g. "1") might follow a longer one (e.g. "12345").
                 # Oh, range(1, K_0_LIMIT+1) means val_p_cand is increasing, so len(s_p_cand) is non-decreasing.
                 # e.g. "9" (len 1) -> "10" (len 2, invalid) -> "11" (len 2).
                 # So, if this one is too long, subsequent ones will be at least as long. So we can break.
                 # This break needs val_p_cand > 1, because "1" is short. What if current P list has long strings?
                 # Better to just 'continue'.
                 continue
            
            s_rev_p_cand = reverse_str(s_p_cand)
            # is_valid_num_str(s_rev_p_cand) is true if s_p_cand is valid.
            val_rev_p_cand = int(s_rev_p_cand)

            prod_contrib = val_p_cand * val_rev_p_cand
            
            if prod_contrib == 0: # Should not happen for val_p_cand >= 1
                continue
            
            if prod_contrib > n_rem : # Optimization: if prod_contrib itself is too large
                # (only if val_p_cand is large, because val_rev_p_cand is tied to val_p_cand's string form)
                # If val_p_cand * val_rev_p_cand > n_rem, then it cannot be a factor.
                # For very large val_p_cand, prod_contrib can exceed n_rem.
                # Example: n_rem=100, val_p_cand=10, val_rev_p_cand=1 (from "01" reversed, but "01" invalid).
                # val_p_cand=10 is invalid itself.
                # val_p_cand=11, val_rev_p_cand=11. prod_contrib=121. If n_rem=100, 121 > 100, skip.
                continue


            if n_rem % prod_contrib == 0:
                new_n_rem = n_rem // prod_contrib
                
                p_list.append(s_p_cand) # Add P_{k+1}
                dfs(new_n_rem, p_list)
                p_list.pop() # Backtrack

                if ans_S is not None: # Solution found in deeper call
                    return
        
    # Initial call to DFS
    dfs(N, [])
    
    if ans_S is None:
        print("-1")
    else:
        print(ans_S)

solve()