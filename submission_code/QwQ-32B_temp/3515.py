from typing import List

class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        sum_single = sum(num for num in nums if num < 10)
        sum_double = sum(num for num in nums if num >= 10)
        return sum_single != sum_double