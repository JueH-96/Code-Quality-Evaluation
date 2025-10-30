from typing import List
from collections import Counter

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        m = max(nums)
        if len(nums) != m + 1:
            return False
        
        counts = Counter(nums)
        
        if counts[m] != 2:
            return False
        
        for num in range(1, m):
            if counts.get(num, 0) != 1:
                return False
        
        return True