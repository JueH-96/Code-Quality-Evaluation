from typing import List

class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_sum = sum(num for num in nums if num < 10)
        double_sum = sum(num for num in nums if num >= 10)
        
        # Alice can win if choosing one group gives a strictly greater sum than the other.
        return single_sum > double_sum or double_sum > single_sum