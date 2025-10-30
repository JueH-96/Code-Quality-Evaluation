from typing import List

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = len(nums) - 1
        base = list(range(1, n)) + [n, n]
        return sorted(nums) == base