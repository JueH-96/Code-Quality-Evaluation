from typing import List
from collections import Counter

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        # Find the maximum element in nums (candidate n)
        n = max(nums)
        
        # base[n] should be of length n + 1
        if len(nums) != n + 1:
            return False
        
        freq = Counter(nums)
        
        # Check that numbers 1 through n-1 appear exactly once
        # and that n appears exactly twice
        for i in range(1, n):
            if freq[i] != 1:
                return False
        if freq[n] != 2:
            return False
        
        # Check that no elements outside the range [1, n] appear
        for key in freq:
            if key < 1 or key > n:
                return False
        
        return True