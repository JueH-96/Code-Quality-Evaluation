from collections import Counter
from typing import List

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        c1 = Counter(nums1)
        c2 = Counter(nums2)
        
        unique1 = set(nums1) - set(nums2)
        unique2 = set(nums2) - set(nums1)
        
        common = set(nums1) & set(nums2)
        
        # Remove half of the common elements from both sets
        for num in common:
            half = (c1[num] + c2[num]) // 2
            c1[num] -= half
            c2[num] -= half
        
        # Add the remaining elements to the result
        result = sum(1 for num in common if c1[num] > 0 or c2[num] > 0)
        
        # Add the unique elements to the result
        result += len(unique1) + len(unique2)
        
        return result