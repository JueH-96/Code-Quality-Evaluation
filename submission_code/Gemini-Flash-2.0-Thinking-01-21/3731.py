from typing import List

class Solution:
    def subarraySum(self, nums: List[int]) -> int:
        # Get the number of elements in the array
        n = len(nums)
        
        # Initialize a variable to store the total sum of all subarrays
        total_sum = 0
        
        # Iterate through each index i from 0 to n-1
        for i in range(n):
            # Calculate the start index for the subarray ending at index i.
            # The rule is start = max(0, i - nums[i]).
            # This ensures the start index is not less than 0.
            start = max(0, i - nums[i])
            
            # Define the subarray for the current index i.
            # The subarray is nums[start ... i], inclusive of both start and i.
            # In Python slicing, nums[a : b] includes elements from index a up to b-1.
            # To include index i, the slice end index should be i + 1.
            current_subarray = nums[start : i + 1]
            
            # Calculate the sum of elements in the current subarray.
            # Using Python's built-in sum function on the slice
            current_subarray_sum = sum(current_subarray)
            
            # Add the sum of the current subarray to the running total sum.
            total_sum += current_subarray_sum
            
        # After iterating through all indices, return the total sum.
        return total_sum