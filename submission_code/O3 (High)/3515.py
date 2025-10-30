from typing import List

class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        # Sum of single–digit numbers (1-9)
        sum_single = sum(x for x in nums if x < 10)
        # Sum of double–digit numbers (10-99)
        sum_double = sum(x for x in nums if x >= 10)
        
        # Alice picks the category whose sum is larger. She wins iff
        # the two sums are different (the larger one will be hers and
        # will be strictly greater than Bob's).
        return sum_single != sum_double