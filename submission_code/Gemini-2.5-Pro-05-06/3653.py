import math
from typing import List # Assuming this is available or pre-imported based on starter code

class Solution:
  def maxSubarraySum(self, nums: List[int], k: int) -> int:
    n = len(nums)
    
    max_s = -math.inf

    min_prefix_sum_for_mod = [math.inf] * k
    min_prefix_sum_for_mod[0] = 0
    
    current_prefix_sum = 0
    for j in range(n):
        current_prefix_sum += nums[j]
        
        # conceptual_idx_end = j + 1
        # current_mod is (conceptual_idx_end % k)
        current_mod = (j + 1) % k
        
        if min_prefix_sum_for_mod[current_mod] != math.inf:
            # current_prefix_sum is P[conceptual_idx_end]
            # min_prefix_sum_for_mod[current_mod] is min(P[conceptual_idx_start])
            # where conceptual_idx_start % k == current_mod
            candidate_sum = current_prefix_sum - min_prefix_sum_for_mod[current_mod]
            if candidate_sum > max_s:
                max_s = candidate_sum
        
        # Update min_prefix_sum_for_mod[current_mod] with current_prefix_sum (P[j+1])
        if current_prefix_sum < min_prefix_sum_for_mod[current_mod]:
            min_prefix_sum_for_mod[current_mod] = current_prefix_sum
            
    return int(max_s)