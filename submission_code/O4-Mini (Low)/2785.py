from typing import List

class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        # find current positions of 1 and n
        pos1 = nums.index(1)
        posn = nums.index(n)
        
        # swaps to bring 1 to front + swaps to bring n to end
        swaps = pos1 + (n - 1 - posn)
        
        # if n is to the left of 1, one of the swaps for moving 1
        # effectively moves n one position further, so we subtract 1
        if posn < pos1:
            swaps -= 1
        
        return swaps