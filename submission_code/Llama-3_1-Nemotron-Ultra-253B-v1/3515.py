from typing import List

class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        sum_single = 0
        sum_double = 0
        for num in nums:
            if 1 <= num <= 9:
                sum_single += num
            else:
                sum_double += num
        can_single = sum_single > sum_double and sum_single > 0
        can_double = sum_double > sum_single and sum_double > 0
        return can_single or can_double