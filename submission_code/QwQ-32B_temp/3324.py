from collections import Counter
from typing import List

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        counts = Counter(nums)
        
        # Check if any element occurs more than twice
        for count in counts.values():
            if count > 2:
                return False
        
        m = len(nums) // 2
        k = len(counts)  # number of unique elements
        c = sum(1 for v in counts.values() if v == 2)  # count of elements with exactly 2 occurrences
        
        return k >= m and c <= m