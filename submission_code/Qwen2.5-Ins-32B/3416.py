from typing import List

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        # Convert each number to a string to easily access digits
        str_nums = [str(num) for num in nums]
        # Length of each number (all have the same length)
        num_length = len(str_nums[0])
        # Initialize the total sum of digit differences
        total_diff = 0
        
        # Iterate over each digit position
        for i in range(num_length):
            # Count the occurrences of each digit at position i
            digit_count = [0] * 10
            for num in str_nums:
                digit_count[int(num[i])] += 1
            
            # Calculate the contribution of this digit position to the total difference
            for count in digit_count:
                total_diff += count * (len(nums) - count)
        
        # Each pair is counted twice, so divide the result by 2
        return total_diff // 2