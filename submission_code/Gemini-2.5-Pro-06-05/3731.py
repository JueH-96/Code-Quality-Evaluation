from typing import List

class Solution:
    def subarraySum(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum_of_subarrays = 0

        # Iterate through each index i from 0 to n-1.
        for i in range(n):
            # For each index i, determine the start of the subarray as per the definition.
            # start = max(0, i - nums[i])
            start_index = max(0, i - nums[i])
            
            # The subarray is defined as nums[start_index ... i].
            # In Python, list slicing is exclusive at the end, so the slice is nums[start_index : i + 1].
            # The built-in sum() function is used to calculate the sum of this subarray's elements.
            current_sum = sum(nums[start_index : i + 1])
            
            # Add the sum of the current subarray to the overall total.
            total_sum_of_subarrays += current_sum
            
        return total_sum_of_subarrays