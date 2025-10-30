from collections import Counter
from typing import List

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = max(nums)
        if len(nums) != n + 1:
            return False
        
        count = Counter(nums)
        
        if count[n] != 2:
            return False
        
        for num in count:
            if num == n:
                continue
            if num < 1 or num > n - 1:
                return False
            if count[num] != 1:
                return False
        
        return True