from typing import List

class Solution:
    def subarraySum(self, nums: List[int]) -> int:
        total_sum = 0
        n = len(nums)
        
        # Iterate through each index i from 0 to n-1
        for i in range(n):
            # Define the start index for the subarray nums[start ... i]
            # start = max(0, i - nums[i])
            # The subarray for index i starts at index max(0, i - nums[i]) and ends at index i (inclusive).
            start = max(0, i - nums[i])
            
            # Extract the subarray for the current index i and calculate its sum
            # In Python slicing, nums[start : i + 1] includes elements from index start up to and including index i.
            current_sum = sum(nums[start : i + 1])
            
            # Add the sum of the current subarray to the total sum
            total_sum += current_sum
            
        # Return the total sum of all elements from the subarrays defined for each index
        return total_sum