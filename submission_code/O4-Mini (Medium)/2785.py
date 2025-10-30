from typing import List

class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        # find positions of 1 and n
        pos1 = nums.index(1)
        posn = nums.index(n)
        # moves to bring 1 to front + moves to bring n to end
        res = pos1 + (n - 1 - posn)
        # if 1 is to the right of n, when we move 1 left past n, n shifts right by 1
        if pos1 > posn:
            res -= 1
        return res