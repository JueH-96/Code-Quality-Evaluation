from typing import List

class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        total = sum(nums)
        sum_single = sum(x for x in nums if x < 10)
        sum_double = sum(x for x in nums if x >= 10)
        
        # Check if choosing single-digit numbers allows Alice to win
        if sum_single > (total - sum_single):
            return True
        # Check if choosing double-digit numbers allows Alice to win
        if sum_double > (total - sum_double):
            return True
        # Neither case works
        return False