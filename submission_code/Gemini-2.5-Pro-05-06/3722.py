import math
from typing import List

class Solution:
  def maxSum(self, nums: List[int], k: int, m: int) -> int:
    n = len(nums)

    # Calculate prefix sums
    # psa[i] stores sum of nums[0...i-1]
    # psa[0] = 0
    psa = [0.0] * (n + 1) # Use float for consistency with -math.inf
    for i in range(n):
        psa[i + 1] = psa[i] + nums[i]

    # dp_prev[i] will store the maximum sum for (j-1) subarrays using nums[0...i-1].
    # Initialize for j=0: max sum of 0 subarrays is 0.
    dp_prev = [0.0] * (n + 1)

    # Iterate for number of subarrays from 1 to k
    for j_count in range(1, k + 1):
        # dp_curr[i] will store the maximum sum of 'j_count' subarrays using nums[0...i-1]
        dp_curr = [-math.inf] * (n + 1)
        
        # best_val_for_prev_sum stores max(dp_prev[p] - psa[p]) over relevant p
        best_val_for_prev_sum = -math.inf
        
        # Smallest number of elements needed for (j_count-1) subarrays, each of length m.
        min_elements_for_prev_subarrays = (j_count - 1) * m
        
        # Smallest number of elements needed for j_count subarrays, each of length m.
        min_elements_for_curr_subarrays = j_count * m

        # Iterate `i` from 1 to n. `i` is the count of elements from `nums` considered (nums[0...i-1]).
        for i in range(1, n + 1):
            # Option 1: The j_count-th subarray does not end at nums[i-1].
            # The sum is taken from the best sum using j_count subarrays from nums[0...i-2].
            # dp_curr[0] is -math.inf (cannot form j_count > 0 subarrays with 0 elements).
            dp_curr[i] = dp_curr[i-1] 

            # Update best_val_for_prev_sum:
            # Consider p = i-m as a potential split point.
            # (j_count-1) subarrays use nums[0...p-1].
            # j_count-th subarray is nums[p...i-1], which has length m.
            p_for_updating_best_val = i - m 
            if p_for_updating_best_val >= min_elements_for_prev_subarrays:
                # Ensure dp_prev[p_for_updating_best_val] is a valid sum
                if dp_prev[p_for_updating_best_val] != -math.inf:
                    current_term = dp_prev[p_for_updating_best_val] - psa[p_for_updating_best_val]
                    best_val_for_prev_sum = max(best_val_for_prev_sum, current_term)

            # Option 2: The j_count-th subarray ends at nums[i-1].
            # This subarray has some start `p_star` (index in `dp_prev` table)
            # and length `i - p_star >= m`.
            # The sum is (dp_prev[p_star] - psa[p_star]) + psa[i].
            # `best_val_for_prev_sum` holds max(dp_prev[p_star] - psa[p_star]).
            if best_val_for_prev_sum != -math.inf:
                # Ensure enough elements are available for j_count subarrays of min length m.
                if i >= min_elements_for_curr_subarrays:
                    candidate_sum = psa[i] + best_val_for_prev_sum
                    dp_curr[i] = max(dp_curr[i], candidate_sum)
        
        # Current dp_curr becomes dp_prev for the next iteration (next j_count)
        dp_prev = dp_curr

    # The final answer is the max sum of k subarrays using up to n elements.
    result = dp_prev[n]
    
    # Constraints k <= floor(n/m) ensure a solution always exists.
    # So, result should not be -math.inf.
    return int(result)