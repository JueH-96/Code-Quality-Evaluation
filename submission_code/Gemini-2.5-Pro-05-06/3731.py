from typing import List

class Solution:
    def subarraySum(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Step 1: Compute prefix sums for efficient subarray sum calculation.
        # prefix_sum_arr[k] stores sum(nums[0]...nums[k-1]).
        # prefix_sum_arr[0] is 0 by convention.
        # The sum of elements from nums[a] to nums[b] (inclusive) 
        # is prefix_sum_arr[b+1] - prefix_sum_arr[a].
        
        prefix_sum_arr = [0] * (n + 1)
        for k in range(n):
            prefix_sum_arr[k+1] = prefix_sum_arr[k] + nums[k]
            
        total_accumulator = 0

        # Step 2: Iterate through each index i to define its associated subarray,
        # calculate its sum, and add to the total_accumulator.
        for i in range(n):
            # For the current index i, the subarray is nums[start_index ... i].
            # The start_index is defined by the problem as max(0, i - nums[i]).
            # Note: nums[i] here is the *value* of the element at index i.
            
            start_index = max(0, i - nums[i])
            
            # The end_index of this subarray (inclusive) is i.
            # Using prefix sums, the sum of nums[start_index ... i] is
            # prefix_sum_arr[i+1] - prefix_sum_arr[start_index].
            # (This corresponds to P[end_inclusive+1] - P[start_inclusive])
            sum_of_defined_subarray = prefix_sum_arr[i+1] - prefix_sum_arr[start_index]
            
            # Add this subarray's sum to the total_accumulator.
            total_accumulator += sum_of_defined_subarray
            
        return total_accumulator