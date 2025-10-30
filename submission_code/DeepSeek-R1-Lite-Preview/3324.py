from typing import List
from collections import Counter

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        n = len(nums)
        k = n // 2
        freq = Counter(nums)
        
        # Check if any element appears more than twice
        for count in freq.values():
            if count > 2:
                return False
        
        # Check if the number of unique elements is at least k
        if len(freq) >= k:
            return True
        else:
            return False