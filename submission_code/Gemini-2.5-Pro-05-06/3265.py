import math
from typing import List

class Solution:
  def maximumSubarraySum(self, nums: List[int], k: int) -> int:
    max_s = -math.inf  # Initialize with negative infinity for comparison
    found_good_subarray = False
    
    # map_val_to_min_Pi stores:
    # key: a value `v` that has appeared in `nums` at some index `i < j`.
    # value: the minimum prefix sum `P[i] = nums[0] + ... + nums[i-1]`
    #        among all such `i` where `nums[i] == v`.
    map_val_to_min_Pi = {}
    
    # current_P stores the prefix sum P[j] = nums[0] + ... + nums[j-1].
    # Initialize P[0] = 0 (sum before the first element).
    current_P = 0
    
    n = len(nums)
    for j in range(n):
        # nums[j] is the current element, potentially the end of a good subarray.
        # We are looking for a subarray nums[i..j] where i < j.
        # Its sum is (nums[0] + ... + nums[j]) - (nums[0] + ... + nums[i-1])
        # which is (P[j] + nums[j]) - P[i].
        # In our terms: (current_P + nums[j]) - map_val_to_min_Pi[nums[i_val]]
        
        val_at_j = nums[j]
        
        # Potential value for nums[i] is target_val_for_i.
        # Condition 1: nums[j] - nums[i] == k  => nums[i] = val_at_j - k
        target_val_for_i_1 = val_at_j - k
        if target_val_for_i_1 in map_val_to_min_Pi:
            Pi = map_val_to_min_Pi[target_val_for_i_1]
            # Sum of nums[i..j]
            current_subarray_sum = (current_P + val_at_j) - Pi
            if current_subarray_sum > max_s:
                max_s = current_subarray_sum
            found_good_subarray = True
        
        # Condition 2: nums[i] - nums[j] == k  => nums[i] = val_at_j + k
        target_val_for_i_2 = val_at_j + k
        if target_val_for_i_2 in map_val_to_min_Pi:
            Pi = map_val_to_min_Pi[target_val_for_i_2]
            # Sum of nums[i..j]
            current_subarray_sum = (current_P + val_at_j) - Pi
            if current_subarray_sum > max_s:
                max_s = current_subarray_sum
            found_good_subarray = True
        
        # Prepare for next iterations: nums[j] can be a starting element nums[i]
        # for a future subarray ending at j' > j.
        # The prefix sum P[j] (which is `current_P` at this point) is associated with nums[j].
        # If nums[j] is already in map, update only if new P[j] is smaller.
        # This ensures we always use the P[i] that maximizes (Sum - P[i]).
        
        # val_to_store_in_map is nums[j]
        # prefix_sum_before_this_val is current_P (which is P[j])
        if nums[j] not in map_val_to_min_Pi or current_P < map_val_to_min_Pi[nums[j]]:
            map_val_to_min_Pi[nums[j]] = current_P
        
        # Update current_P from P[j] to P[j+1] for the next iteration (j -> j+1)
        current_P += nums[j]
            
    if not found_good_subarray:
        return 0
    else:
        # max_s holds the maximum sum found. This could be negative.
        # Python integers handle large numbers, so int() cast is mostly for type explicitness
        # if -math.inf was never replaced (though found_good_subarray guards this).
        return int(max_s)