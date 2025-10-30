from typing import List
from collections import Counter

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        n = len(nums)
        half = n // 2
        freq = Counter(nums)
        
        # If any element appears more than twice, we can't avoid duplicates in one half.
        if any(count > 2 for count in freq.values()):
            return False
        
        # If we have at least `half` distinct elements, we can pick half distinct for nums1
        # and place the remaining copies (or other distincts) in nums2.
        return len(freq) >= half