import math

class Solution:
  def maximumStrength(self, nums: List[int], k: int) -> int:
    n = len(nums)
    neg_inf = -float('inf')

    # ps[i] stores sum of nums[0...i-1] (i.e., sum of the first i elements)
    # ps[0] = 0
    # ps[idx] = sum(nums[0...idx-1])
    ps = [0] * (n + 1)
    for i in range(n):
        ps[i + 1] = ps[i] + nums[i]

    # dp_prev[i] stores max strength using (j_count-1) subarrays,
    # considering the first i elements (nums[0...i-1]).
    # Initialize for j_count = 0 (0 subarrays): strength is 0.
    dp_prev = [0] * (n + 1) 

    # Iterate for j_count from 1 to k (number of subarrays)
    for j_count in range(1, k + 1):
        # Calculate coefficient for the j_count-th subarray
        # Formula: (-1)^(j_count+1) * (k - j_count + 1)
        coeff_val = (k - j_count + 1)
        if j_count % 2 == 0: # j_count is even, term is negative
            coeff = -coeff_val
        else: # j_count is odd, term is positive
            coeff = coeff_val
        
        # dp_curr[i] will store max strength using j_count subarrays,
        # considering the first i elements (nums[0...i-1]).
        # Initialize with neg_inf. dp_curr[0] (j_count subarrays from 0 elements) is neg_inf.
        dp_curr = [neg_inf] * (n + 1) 
        
        # max_term_to_add = max_{0 <= p_table_idx < i_idx} (dp_prev[p_table_idx] - ps[p_table_idx] * coeff)
        # This is optimized by updating it iteratively.
        # p_table_idx is the DP table index, corresponding to considering p_table_idx elements.
        max_term_to_add = neg_inf
        
        # i_idx represents the number of elements considered from nums (nums[0...i_idx-1]).
        # Loop i_idx from 1 to n.
        for i_idx in range(1, n + 1):
            # Option 1: The j_count subarrays are formed from nums[0...i_idx-2] (first i_idx-1 elements).
            # This means nums[i_idx-1] is not used, or not part of the j_count-th subarray ending here.
            # The value comes from dp_curr[i_idx-1] (already computed for current j_count).
            # dp_curr[0] is neg_inf, so if i_idx=1, opt1_not_ending_at_current_num = neg_inf.
            opt1_not_ending_at_current_num = dp_curr[i_idx-1]

            # Option 2: nums[i_idx-1] is the end of the j_count-th subarray.
            # This subarray starts at some index p_start_in_nums (0-indexed in nums array).
            # The previous (j_count-1) subarrays are formed from nums[0...p_start_in_nums-1].
            # This corresponds to dp_prev[p_start_in_nums] (using p_start_in_nums elements).
            # The prefix sum for this is ps[p_start_in_nums].
            
            # Update max_term_to_add:
            # The new candidate for p_table_idx in `max_term_to_add` is `i_idx-1`.
            # This corresponds to the case where the j_count-th subarray starts at `nums[i_idx-1]`.
            # (i.e., the j_count-th subarray is just `nums[i_idx-1 ... i_idx-1]`).
            # The (j_count-1) subarrays are formed from `nums[0...i_idx-2]`.
            # This state is `dp_prev[i_idx-1]` (using `i_idx-1` elements).
            # The prefix sum for the `ps[p_table_idx]` part is `ps[i_idx-1]`.
            # This update is valid only if `i_idx-1 >= j_count-1` (enough elements for j_count-1 subarrays).
            if i_idx - 1 >= j_count - 1:
                # `dp_prev[i_idx-1]` must not be `neg_inf`.
                # For `j_count = 1`, `dp_prev[i_idx-1]` is `0`, which is not `neg_inf`.
                if dp_prev[i_idx-1] != neg_inf:
                    current_cand_for_max_term = dp_prev[i_idx-1] - ps[i_idx-1] * coeff
                    if max_term_to_add == neg_inf or current_cand_for_max_term > max_term_to_add:
                        max_term_to_add = current_cand_for_max_term
            
            opt2_ending_at_current_num = neg_inf
            # If max_term_to_add is still neg_inf, it means no valid way to form Option 2.
            # This happens if i_idx < j_count.
            if max_term_to_add != neg_inf:
                # ps[i_idx] is sum of nums[0...i_idx-1]
                opt2_ending_at_current_num = max_term_to_add + ps[i_idx] * coeff
            
            # dp_curr[i_idx] is the maximum of these two options.
            # If i_idx < j_count, both opt1 and opt2 will resolve to neg_inf.
            # Thus, dp_curr[i_idx] will correctly be neg_inf.
            current_max_strength_at_i_idx = neg_inf
            if opt1_not_ending_at_current_num != neg_inf:
                current_max_strength_at_i_idx = opt1_not_ending_at_current_num
            
            if opt2_ending_at_current_num != neg_inf:
                if current_max_strength_at_i_idx == neg_inf or \
                   opt2_ending_at_current_num > current_max_strength_at_i_idx:
                    current_max_strength_at_i_idx = opt2_ending_at_current_num
            
            dp_curr[i_idx] = current_max_strength_at_i_idx
            
        # Current dp_curr becomes dp_prev for the next j_count iteration.
        dp_prev = dp_curr 

    # After all iterations, dp_prev holds the results for j_count = k.
    # The answer is for n elements.
    ans = dp_prev[n]
    
    return ans