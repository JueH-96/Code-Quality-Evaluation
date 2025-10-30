from collections import Counter
from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        count = Counter(nums)
        original = count.get(k, 0)
        max_val = original  # Default to original count if no other values
        
        for v in count:
            if v != k:
                current = count[v] + original
                if current > max_val:
                    max_val = current
        
        return max_val