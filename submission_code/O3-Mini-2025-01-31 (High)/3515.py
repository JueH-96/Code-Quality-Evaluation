from typing import List

class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        # Sum of all single-digit numbers (< 10)
        single_digit_sum = sum(x for x in nums if x < 10)
        # Sum of all double-digit numbers (>= 10)
        double_digit_sum = sum(x for x in nums if x >= 10)
        
        # Alice can win if by picking the numbers from whichever category is higher, 
        # her sum is strictly greater than Bob's (the sum of the remaining numbers).
        return single_digit_sum > double_digit_sum or double_digit_sum > single_digit_sum