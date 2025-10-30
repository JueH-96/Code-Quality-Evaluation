from typing import List

class Solution:
  def sumOfPowers(self, nums: List[int], k: int) -> int:
    n = len(nums)
    MOD = 10**9 + 7

    nums.sort()

    distinct_differences = set()
    for i in range(n):
        for j in range(i + 1, n):
            diff = nums[j] - nums[i]
            # Power is defined as min *absolute* difference.
            # Since nums is sorted, nums[j]-nums[i] is non-negative.
            # We only care about positive differences for the sum P(S) = sum_{x>=1} count_ge(x) formula.
            # If diff is 0, it means power could be 0. Such subsequences add 0 to sum.
            if diff > 0:
                 distinct_differences.add(diff)
    
    sorted_diffs = sorted(list(distinct_differences))

    if not sorted_diffs and k > 0: # Only if all elements are same or n < 2 (but n>=2 constraint)
                                  # If all elements are same, all powers are 0.
        return 0

    # dp[length][last_idx_in_nums]
    # This table will be reused for each v_diff.
    dp = [[0] * n for _ in range(k + 1)]

    total_sum_of_powers = 0
    last_processed_d_val = 0
    
    for v_diff in sorted_diffs:
        # Calculate count_ge(v_diff): number of subsequences of length k
        # with minimum difference >= v_diff.

        # Base case: subsequences of length 1
        for i in range(n):
            dp[1][i] = 1
        
        for length in range(2, k + 1):
            # Zero out dp[length] row before computing for current v_diff and length
            for i in range(n): 
                dp[length][i] = 0

            current_sum_prev_len = 0 
            left_ptr = 0 # Pointer for elements in dp[length-1]
            
            # right_ptr is the index of the current element being added to subsequence.
            # A subsequence of `length` must end at index at least `length-1`.
            # e.g., for length=2, earliest nums[right_ptr] is nums[1] (using nums[0], nums[1]).
            for right_ptr in range(length - 1, n):
                # Advance left_ptr: sum dp[length-1][left_ptr] for valid previous elements.
                # A previous element nums[left_ptr] is valid if nums[right_ptr] - nums[left_ptr] >= v_diff.
                # Also, left_ptr must be < right_ptr to pick distinct positions for subsequence elements.
                while left_ptr < right_ptr and (nums[right_ptr] - nums[left_ptr] >= v_diff):
                    current_sum_prev_len = (current_sum_prev_len + dp[length-1][left_ptr]) % MOD
                    left_ptr += 1
                
                # current_sum_prev_len now contains sum of dp[length-1][x] for x in [0...left_ptr-1].
                # These are all x such that nums[right_ptr] - nums[x] >= v_diff (verified by logic in thought process).
                dp[length][right_ptr] = current_sum_prev_len
        
        # Total count of subsequences of length k with min_diff >= v_diff
        count_for_this_v_diff = 0
        # Summing up dp[k][i] for all i. dp[k][i] is 0 if i < k-1 due to loop range for right_ptr.
        for i in range(k - 1, n): 
            count_for_this_v_diff = (count_for_this_v_diff + dp[k][i]) % MOD
        
        # Add to total_sum_of_powers: count_ge(v_diff) * (v_diff - last_processed_d_val)
        term_value = (count_for_this_v_diff * (v_diff - last_processed_d_val)) % MOD
        total_sum_of_powers = (total_sum_of_powers + term_value) % MOD
        last_processed_d_val = v_diff
            
    return total_sum_of_powers