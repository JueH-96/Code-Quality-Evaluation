from typing import List

class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        pos1 = nums.index(1)
        posN = nums.index(n)
        
        # If 1 is after n, we need to adjust the count since moving 1 to the front will bring n closer to the end
        if pos1 > posN:
            return pos1 + (n - posN - 1) - 1
        else:
            return pos1 + (n - posN - 1)