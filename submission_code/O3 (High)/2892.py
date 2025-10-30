from typing import List
from collections import Counter

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        # Candidate n is the maximum element in nums
        n = max(nums)
        
        # base[n] has length n + 1
        if len(nums) != n + 1:
            return False
        
        freq = Counter(nums)
        
        # n must appear exactly twice
        if freq[n] != 2:
            return False
        
        # Every integer from 1 to n-1 must appear exactly once
        for x in range(1, n):
            if freq[x] != 1:
                return False
        
        # No other numbers are allowed
        if len(freq) != n:      # we should have keys 1..n   (n distinct keys)
            return False
        
        return True