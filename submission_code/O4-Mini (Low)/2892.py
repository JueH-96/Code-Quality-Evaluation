from typing import List
from collections import Counter

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        # The maximum element in nums would be our candidate n
        n = max(nums)
        # The good array base[n] has length n + 1
        if len(nums) != n + 1:
            return False
        
        cnt = Counter(nums)
        # For a good array:
        #  - each integer from 1 to n-1 must appear exactly once
        #  - integer n must appear exactly twice
        # Also there must be no other integers outside 1..n
        for x in range(1, n):
            if cnt.get(x, 0) != 1:
                return False
        if cnt.get(n, 0) != 2:
            return False
        
        # Finally ensure no extra keys
        # (all keys are between 1 and n, inclusive)
        if any(k < 1 or k > n for k in cnt):
            return False
        
        return True