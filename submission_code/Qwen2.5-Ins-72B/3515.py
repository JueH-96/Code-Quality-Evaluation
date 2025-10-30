from typing import List

class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit_sum = sum(num for num in nums if num < 10)
        double_digit_sum = sum(num for num in nums if num >= 10)
        
        total_sum = sum(nums)
        
        # Alice wins if the sum of single-digit numbers is greater than the sum of double-digit numbers
        if single_digit_sum > total_sum - single_digit_sum:
            return True
        
        # Alice wins if the sum of double-digit numbers is greater than the sum of single-digit numbers
        if double_digit_sum > total_sum - double_digit_sum:
            return True
        
        return False