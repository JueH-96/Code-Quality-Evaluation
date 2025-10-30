from typing import List

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        # Convert all integers to strings for easier comparison
        nums = [str(num) for num in nums]
        
        # Initialize the total sum of digit differences
        total_sum = 0
        
        # Calculate the length of the integers (all integers have the same length)
        num_length = len(nums[0])
        
        # Iterate over all pairs of integers
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                # Calculate the digit difference between the current pair of integers
                digit_diff = sum(1 for k in range(num_length) if nums[i][k] != nums[j][k])
                
                # Add the digit difference to the total sum
                total_sum += digit_diff
        
        return total_sum