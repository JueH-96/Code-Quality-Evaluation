from collections import defaultdict
from typing import List

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        freq1 = defaultdict(int)
        for num in nums1:
            freq1[num] += 1
        freq2 = defaultdict(int)
        for num in nums2:
            freq2[num] += 1
        
        common = set(freq1.keys()) & set(freq2.keys())
        u1 = sum(1 for k in freq1 if k not in common)
        u2 = sum(1 for k in freq2 if k not in common)
        c = len(common)
        
        x = min(u1, n // 2)
        y = min(u2, n // 2)
        a_remaining = (n // 2) - x
        b_remaining = (n // 2) - y
        common_in_A_or_B = min(c, a_remaining + b_remaining)
        
        return x + y + common_in_A_or_B