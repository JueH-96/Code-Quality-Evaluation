from typing import List

class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        # Sum of single-digit (1-9) and double-digit (10-99) numbers
        sum_single = sum(x for x in nums if x <= 9)
        sum_double = sum(x for x in nums if x >= 10)
        
        # Alice will pick whichever category gives her a strictly greater sum
        # If the sums are equal, she cannot win
        return sum_single != sum_double