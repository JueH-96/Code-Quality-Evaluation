from typing import List

class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        # Initialize a variable to store the maximum value
        max_val = 0
        
        # Iterate through the array from left to right
        i = 0
        while i < len(nums) - 1:
            # If the current element is less than or equal to the next element, merge them
            if nums[i] <= nums[i + 1]:
                # Update the current element with the sum of the current and next elements
                nums[i] += nums[i + 1]
                # Remove the next element from the array
                nums.pop(i + 1)
            else:
                # If the current element is greater than the next element, move to the next element
                i += 1
        
        # Update the maximum value
        max_val = max(nums)
        
        return max_val