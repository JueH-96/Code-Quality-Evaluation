from typing import List

class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        sum_single = 0
        sum_double = 0
        for num in nums:
            if num <= 9:
                sum_single += num
            else:
                sum_double += num
        return sum_single != sum_double