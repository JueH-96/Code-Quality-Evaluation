from typing import List
from collections import Counter

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        max_num = max(nums)
        n = max_num
        # Check if the length of nums is n + 1
        if len(nums) != n + 1:
            return False
        # Count the frequency of each number
        count = Counter(nums)
        # Check that the maximum number appears exactly twice
        if count[n] != 2:
            return False
        # Check that each number from 1 to n-1 appears exactly once
        for i in range(1, n):
            if count.get(i, 0) != 1:
                return False
        return True