import math
from typing import List

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        
        # This map stores for each number `v` encountered, the minimum prefix sum
        # leading up to an index `i` where `nums[i] == v`.
        # Key: number `v` from nums.
        # Value: minimum prefix sum `prefix[i]`.
        min_prefix_sum_for_val = {}
        
        # We use a sentinel for max_sum because the result can be negative.
        # The flag handles the case where no good subarray exists.
        max_s = -math.inf
        found_good_subarray = False
        
        # `current_prefix_sum` holds the running prefix sum. At the start of the
        # loop for index `j`, it equals sum(nums[0...j-1]).
        current_prefix_sum = 0
        
        for j in range(len(nums)):
            # First, consider nums[j] as a potential starting point for a future subarray.
            # The prefix sum for a subarray starting at index `j` is `current_prefix_sum`.
            # We record this prefix sum against the value `nums[j]`.
            val_at_j = nums[j]
            if val_at_j not in min_prefix_sum_for_val:
                min_prefix_sum_for_val[val_at_j] = current_prefix_sum
            else:
                # If we've seen this value before, keep the smaller prefix sum to maximize
                # the potential subarray sum later.
                min_prefix_sum_for_val[val_at_j] = min(min_prefix_sum_for_val[val_at_j], current_prefix_sum)
            
            # The sum of elements up to and including `j` is the `prefix_sum[j+1]`.
            sum_up_to_j = current_prefix_sum + val_at_j
            
            # Now, consider nums[j] as the ending point of a subarray. We look for a
            # starting value `v` that satisfies `v = nums[j] +/- k`.
            target1 = val_at_j - k
            target2 = val_at_j + k
            
            # Case 1: Check for a start value of `target1`.
            if target1 in min_prefix_sum_for_val:
                found_good_subarray = True
                # sum = (sum up to j) - (sum before start)
                start_prefix_sum = min_prefix_sum_for_val[target1]
                subarray_sum = sum_up_to_j - start_prefix_sum
                max_s = max(max_s, subarray_sum)
            
            # Case 2: Check for a start value of `target2`.
            if target2 in min_prefix_sum_for_val:
                found_good_subarray = True
                start_prefix_sum = min_prefix_sum_for_val[target2]
                subarray_sum = sum_up_to_j - start_prefix_sum
                max_s = max(max_s, subarray_sum)
            
            # Update the prefix sum for the next iteration.
            current_prefix_sum += val_at_j
            
        if not found_good_subarray:
            return 0
        else:
            return int(max_s)