import math
from typing import List

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Step 1: Calculate prefix sums
        # prefix_sums[x] stores the sum of nums[0]...nums[x-1]
        # prefix_sums[0] is 0 (sum of an empty prefix)
        prefix_sums = [0] * (n + 1)
        for i in range(n):
            prefix_sums[i+1] = prefix_sums[i] + nums[i]
        
        # Step 2: Initialize maximum sum found so far.
        # Use float('-inf') to correctly handle cases where all good subarray sums are negative.
        # If no good subarrays are found, it will remain float('-inf') and we'll return 0.
        max_good_subarray_sum = float('-inf')
        
        # min_prefix_val_map: A dictionary where keys are values from nums[] elements
        # and values are the minimum prefix_sums[index] encountered so far for that element value.
        # Specifically, for a key `val`, min_prefix_val_map[val] stores
        # min(prefix_sums[p]) for all p <= current_j such that nums[p] == val.
        # The update order ensures that when checking for a subarray ending at `j`,
        # the map only contains values for `i < j`.
        min_prefix_val_map = {}
        
        # Step 3: Iterate through the array to find good subarrays
        # `j` represents the current end index of a potential good subarray (nums[j])
        for j in range(n):
            current_val = nums[j]
            # `current_prefix_sum_ending_at_j` is the sum of elements from nums[0] to nums[j] (inclusive)
            current_prefix_sum_ending_at_j = prefix_sums[j+1] 
            
            # Case 1: Look for a starting element `nums[i]` such that `nums[i] = current_val - k`
            target1 = current_val - k
            if target1 in min_prefix_val_map:
                # If target1 is in the map, it means we have found an index `i` (where i < j)
                # such that nums[i] == target1.
                # `min_prefix_val_map[target1]` holds the smallest `prefix_sums[i]` for that target1,
                # thus maximizing the subarray sum `prefix_sums[j+1] - prefix_sums[i]`.
                val_at_i_prefix_sum = min_prefix_val_map[target1]
                current_subarray_sum = current_prefix_sum_ending_at_j - val_at_i_prefix_sum
                max_good_subarray_sum = max(max_good_subarray_sum, current_subarray_sum)
            
            # Case 2: Look for a starting element `nums[i]` such that `nums[i] = current_val + k`
            target2 = current_val + k
            if target2 in min_prefix_val_map:
                # Same logic as Case 1
                val_at_i_prefix_sum = min_prefix_val_map[target2]
                current_subarray_sum = current_prefix_sum_ending_at_j - val_at_i_prefix_sum
                max_good_subarray_sum = max(max_good_subarray_sum, current_subarray_sum)
            
            # After processing `nums[j]` as an end element, add `nums[j]` to the map
            # for potential use as a starting element `i'` for *future* `j' > j`.
            # We store `prefix_sums[j]`, which is the sum of `nums[0]`...`nums[j-1]`.
            # This is the correct prefix sum to use if `nums[j]` becomes `nums[i']` (the start of a subarray).
            prefix_sum_before_current_val = prefix_sums[j] 
            min_prefix_val_map[current_val] = min(
                min_prefix_val_map.get(current_val, float('inf')), 
                prefix_sum_before_current_val
            )
            
        # Step 4: Return the result.
        # If max_good_subarray_sum is still float('-inf'), no good subarrays were found.
        if max_good_subarray_sum == float('-inf'):
            return 0
        else:
            return max_good_subarray_sum