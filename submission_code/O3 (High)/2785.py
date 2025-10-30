from typing import List

class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        
        # positions (0-indexed) of 1 and n in the permutation
        pos1 = nums.index(1)
        posn = nums.index(n)
        
        # If 1 is left of n, their moves are independent.
        # Otherwise they cross once, so one swap is saved.
        if pos1 < posn:
            return pos1 + (n - 1 - posn)
        else:
            return pos1 + (n - 2 - posn)