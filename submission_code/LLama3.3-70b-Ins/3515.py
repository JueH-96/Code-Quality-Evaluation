from typing import List

class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        # Calculate the sum of all numbers
        total_sum = sum(nums)
        
        # Calculate the sum of single-digit numbers
        single_digit_sum = sum(num for num in nums if num < 10)
        
        # Calculate the sum of double-digit numbers
        double_digit_sum = sum(num for num in nums if num >= 10)
        
        # Check if Alice can win by choosing single-digit numbers
        if single_digit_sum > total_sum - single_digit_sum:
            return True
        
        # Check if Alice can win by choosing double-digit numbers
        if double_digit_sum > total_sum - double_digit_sum:
            return True
        
        # If Alice cannot win by choosing either single-digit or double-digit numbers, return False
        return False