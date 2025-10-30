from typing import List
from collections import Counter

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        # Find n as the maximum value in nums
        n = max(nums)
        
        # Check if the length of nums is exactly n + 1
        if len(nums) != n + 1:
            return False
        
        # Count the frequency of each number in nums
        freq = Counter(nums)
        
        # Check frequencies for numbers from 1 to n-1
        for num in range(1, n):
            if freq.get(num, 0) != 1:
                return False
        
        # Check frequency for n (should appear exactly twice)
        if freq.get(n, 0) != 2:
            return False
        
        return True