from typing import List
from collections import Counter

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        # The candidate n is the maximum element in nums
        n = max(nums)
        # A "good" array must have exactly n+1 elements
        if len(nums) != n + 1:
            return False
        
        count = Counter(nums)
        # n must appear exactly twice
        if count[n] != 2:
            return False
        # Each integer from 1 to n-1 must appear exactly once
        for x in range(1, n):
            if count[x] != 1:
                return False
        
        return True