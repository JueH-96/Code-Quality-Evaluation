from typing import List

class Solution:
    def subarraySum(self, nums: List[int]) -> int:
        total_sum = 0
        n = len(nums)

        # Iterate through each index 'i' of the array
        for i in range(n):
            # Calculate the 'start' index for the current subarray as per the problem definition.
            # start = max(0, i - nums[i])
            # This ensures the subarray never starts before index 0.
            start_index = max(0, i - nums[i])
            
            # Extract the subarray from 'start_index' up to and including 'i'.
            # Python slicing [start:end] includes elements from 'start' up to (but not including) 'end'.
            # So, to include nums[i], the 'end' for the slice should be 'i + 1'.
            current_subarray = nums[start_index : i + 1]
            
            # Sum the elements of this specific subarray.
            current_subarray_sum = sum(current_subarray)
            
            # Add the sum of the current subarray to the overall total sum.
            total_sum += current_subarray_sum
            
        # Return the final accumulated total sum.
        return total_sum