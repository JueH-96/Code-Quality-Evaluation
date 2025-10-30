import collections

class Solution:
  def makeStringGood(self, s: str) -> int:
    n = len(s)
    
    # Optimized character counts storage using a list/array of size 26
    char_counts_arr = [0] * 26
    for char_val in s:
        char_counts_arr[ord(char_val) - ord('a')] += 1

    min_total_ops = n # Upper bound: delete all characters (empty string is good)

    # Iterate over all possible target frequencies K.
    # K can range from 1 up to N (e.g. s="aaaa", K=4).
    # K can also be N+1 (e.g. s="a", make "aa", K=2, N=1).
    # Max K could be up to 2N based on D*K < 2N.
    # However, N*alphabet_size*N_K_range is too slow if N_K_range is N.
    # For typical contest limits, N_K_range might be smaller, or N large implies small K_optimal.
    # A common range for K in similar problems is up to N or N + alphabet_size.
    # Let's iterate K up to N + 1. If K is larger, it implies many insertions.
    # For N=2*10^4, N+1 is too large.
    # Given constraints, maybe max K to check is not that large.
    # If N is large, solution with K > (e.g. 100) might already cost > N.
    # Let's try K up to a limit like 200, or slightly more than N for small N.
    # Max K to check: N + 2 (covers N, N+1, small margin)
    # Let's use a K_limit: min(N + 5, X) where X could be e.g. 150-200 for Python.
    # The problem likely means K_max_iter is small if N is large, or N is small if K_max_iter is large.
    # The problem might pass with K_max = N+1 with specific test case structures.
    # For this implementation, let's use N + 2 as upper bound for K loop to be safe for K=N+1 cases.
    # If this TLEs, a fixed smaller K_limit is needed.
    k_loop_limit = n + 2 
    # A more aggressive K_limit for very large N, maybe 200 if N > 200, else N+2
    # if n > 200:
    #    k_loop_limit = 200 # Heuristic limit often used in Python contests for such loops
    
    # dp_char_plus_1[j] = min cost for suffix of alphabet from char_idx+1 to 'z',
    #                     given j chars were passed from char_idx to char_idx+1.
    dp_char_plus_1 = [float('inf')] * (n + 1)
    dp_char_plus_1[0] = 0 # Base case for imaginary char after 'z': 0 cost if 0 chars passed.
                          # Infinite cost if >0 chars passed (handled by logic for 'z').

    for k_target_freq in range(1, k_loop_limit):
        # memo_dp_vals_for_char_plus_1 will store dp table for char_idx+1
        memo_dp_vals_for_char_plus_1 = [float('inf')] * (n + 1)
        memo_dp_vals_for_char_plus_1[0] = 0 # Base case for char_idx = 26 (beyond 'z')

        for char_idx in range(25, -1, -1): # From 'z' (25) down to 'a' (0)
            dp_curr_char = [float('inf')] * (n + 1) # dp table for current char_idx
            
            min_prefix_sum_for_next_dp = [float('inf')] * (n + 1)
            if n >= 0: # Ensure n+1 list has at least one element
                min_prefix_sum_for_next_dp[0] = memo_dp_vals_for_char_plus_1[0]
                for i in range(1, n + 1):
                    min_prefix_sum_for_next_dp[i] = min(min_prefix_sum_for_next_dp[i-1], memo_dp_vals_for_char_plus_1[i])

            actual_initial_char_count = char_counts_arr[char_idx]

            for carry_in in range(n + 1): 
                current_char_stock = actual_initial_char_count + carry_in
                # current_char_stock is sum of initial_counts[0...char_idx], so it's <= N.
                # No explicit cap current_char_stock = min(current_char_stock, n) needed.

                # Option 1: Make count of char_idx equal to k_target_freq
                cost1 = float('inf')
                if current_char_stock >= k_target_freq:
                    cost_for_surplus_handling = current_char_stock - k_target_freq
                    future_contribution = float('inf')
                    if char_idx == 25: # 'z', cannot change to next. All surplus must be deleted. k_next = 0.
                        future_contribution = memo_dp_vals_for_char_plus_1[0]
                    else:
                        limit_k_next = current_char_stock - k_target_freq # Max items to pass
                        # limit_k_next cannot exceed N since current_char_stock <= N.
                        if limit_k_next >= 0: # Valid index check
                           future_contribution = min_prefix_sum_for_next_dp[limit_k_next]
                    
                    if future_contribution != float('inf'): # Check if path is valid
                        cost1 = cost_for_surplus_handling + future_contribution
                else: # current_char_stock < k_target_freq
                    cost_for_insertions = k_target_freq - current_char_stock
                    future_contribution = memo_dp_vals_for_char_plus_1[0] # k_next is 0
                    if future_contribution != float('inf'):
                        cost1 = cost_for_insertions + future_contribution

                # Option 2: Make count of char_idx equal to 0
                cost_for_zeroing_out = current_char_stock
                future_contribution_opt2 = float('inf')
                if char_idx == 25: # 'z', all stock must be deleted. k_next = 0.
                    future_contribution_opt2 = memo_dp_vals_for_char_plus_1[0]
                else:
                    limit_k_next_opt2 = current_char_stock # Max items to pass
                    # limit_k_next_opt2 cannot exceed N.
                    if limit_k_next_opt2 >=0: # Valid index check
                        future_contribution_opt2 = min_prefix_sum_for_next_dp[limit_k_next_opt2]
                
                cost2 = float('inf')
                if future_contribution_opt2 != float('inf'): # Check if path is valid
                    cost2 = cost_for_zeroing_out + future_contribution_opt2
                
                # Cap dp states at min_total_ops (or N) to prune expensive paths early.
                # This is a small optimization, not strictly necessary for correctness here.
                # Can use N+1 as a practical upper bound for dp states if required.
                current_ops_for_state = min(cost1, cost2)
                if current_ops_for_state < dp_curr_char[carry_in]:
                     dp_curr_char[carry_in] = current_ops_for_state
            
            memo_dp_vals_for_char_plus_1 = dp_curr_char 

        if memo_dp_vals_for_char_plus_1[0] < min_total_ops: # Result for this K is dp['a'][0]
            min_total_ops = memo_dp_vals_for_char_plus_1[0]
            
    return min_total_ops