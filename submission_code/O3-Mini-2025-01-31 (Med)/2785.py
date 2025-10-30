from typing import List

class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        pos1 = nums.index(1)
        posn = nums.index(n)
        # If the index of 1 is to the right of n, then when we move 1 left the position of n effectively shifts one slot to the left.
        # Therefore, we subtract one from the total.
        operations = pos1 + (n - 1 - posn)
        if pos1 > posn:
            operations -= 1
        return operations