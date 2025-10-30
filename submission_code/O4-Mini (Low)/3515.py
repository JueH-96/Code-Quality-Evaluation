from typing import List

class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        # Sum of single-digit numbers (1 through 9)
        sum_single = sum(x for x in nums if x < 10)
        # Sum of double-digit numbers (10 through 99)
        sum_double = sum(x for x in nums if x >= 10)
        # Total sum of all numbers
        total = sum_single + sum_double
        
        # If Alice picks singles, she wins if 2 * sum_single > total
        if 2 * sum_single > total:
            return True
        # If Alice picks doubles, she wins if 2 * sum_double > total
        if 2 * sum_double > total:
            return True
        
        # Otherwise she cannot win
        return False