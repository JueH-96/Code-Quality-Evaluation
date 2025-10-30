from typing import List

class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        sum_single_digit = sum(num for num in nums if num < 10)
        sum_double_digit = sum(num for num in nums if num >= 10)
        
        total_sum = sum(nums)
        
        return (sum_single_digit > total_sum - sum_single_digit) or (sum_double_digit > total_sum - sum_double_digit)