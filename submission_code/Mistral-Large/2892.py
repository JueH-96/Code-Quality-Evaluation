from typing import List
from collections import Counter

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = max(nums)
        if len(nums) != n + 1:
            return false

        count = Counter(nums)
        for i in range(1, n):
            if count[i] != 1:
                return false
        if count[n] != 2:
            return false

        return true