from typing import List

class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return 0
        pos1 = nums.index(1)
        posN = nums.index(n)
        
        if pos1 == 0 and posN == n - 1:
            return 0
        
        if pos1 < posN:
            return pos1 + (n - 1 - posN)
        else:
            return pos1 + (n - 1 - (posN + 1))