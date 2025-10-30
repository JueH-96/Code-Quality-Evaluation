from typing import List

class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        # Calculate the difference between the two arrays
        diff = [a - b for a, b in zip(nums, target)]
        
        # Initialize the result (number of operations) and the current sum
        res, curr_sum = 0, 0
        
        # Iterate over the differences
        for num in diff:
            # Update the current sum
            curr_sum += num
            
            # If the current sum is positive, increment the result and reset the current sum
            if curr_sum > 0:
                res += curr_sum
                curr_sum = 0
            # If the current sum is negative, increment the result and reset the current sum
            elif curr_sum < 0:
                res -= curr_sum
                curr_sum = 0
        
        return res