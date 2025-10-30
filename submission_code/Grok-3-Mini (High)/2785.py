from typing import List

class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        pos1 = nums.index(1)
        posn = nums.index(n)
        dist1 = pos1
        distn = (n - 1) - posn
        ans = dist1 + distn
        if pos1 > posn:
            ans -= 1
        return ans