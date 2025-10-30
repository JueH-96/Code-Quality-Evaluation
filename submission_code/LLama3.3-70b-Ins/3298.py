from typing import List

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        # Sort the array in ascending order
        nums.sort()
        
        # Initialize variables to keep track of the maximum length and current length of consecutive elements
        max_length = 1
        current_length = 1
        
        # Iterate through the sorted array
        for i in range(1, len(nums)):
            # If the current element is consecutive to the previous element, increment the current length
            if nums[i] - nums[i - 1] <= 1:
                current_length += 1
            # If the current element is not consecutive to the previous element, update the maximum length and reset the current length
            else:
                max_length = max(max_length, current_length)
                current_length = 1
        
        # Update the maximum length one last time
        max_length = max(max_length, current_length)
        
        return max_length