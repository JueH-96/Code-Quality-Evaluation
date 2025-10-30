from typing import List

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        # If nums is good, max(nums) gives candidate n.
        # base[n] should have length n+1, contain each of 1..n-1 exactly once, and n exactly twice.
        n = max(nums)
        # Check length
        if len(nums) != n + 1:
            return False
        
        # Count frequencies
        freq = [0] * (n + 1)
        for x in nums:
            # x must be in [1..n]
            if x < 1 or x > n:
                return False
            freq[x] += 1
        
        # Check 1..n-1 each appear exactly once
        for i in range(1, n):
            if freq[i] != 1:
                return False
        
        # Check n appears exactly twice
        return freq[n] == 2