from typing import List

class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        pos1 = nums.index(1)
        posn = nums.index(n)
        # Moves to bring 1 to front + moves to bring n to end
        ans = pos1 + (n - 1 - posn)
        # If 1 passes over n when moving, one swap is effectively shared
        if pos1 > posn:
            ans -= 1
        return ans