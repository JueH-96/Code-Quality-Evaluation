from typing import List

class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        pos_1 = nums.index(1)
        pos_n = nums.index(n)
        return pos_1 + (n - 1 - pos_n) - (1 if pos_1 > pos_n else 0)