from typing import List

class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        # Sum of single-digit numbers (1 through 9)
        sum_single = sum(x for x in nums if x < 10)
        # Sum of double-digit numbers (10 through 99)
        sum_double = sum(x for x in nums if x >= 10)
        # Alice can choose whichever category gives her a strictly greater sum
        return sum_single > sum_double or sum_double > sum_single