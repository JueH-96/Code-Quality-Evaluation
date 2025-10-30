from typing import List

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        half_n = n // 2
        set1 = set(nums1)
        set2 = set(nums2)
        
        common_elements = set1.intersection(set2)
        unique1 = set1 - common_elements
        unique2 = set2 - common_elements
        
        max_unique1 = min(len(unique1), half_n)
        max_unique2 = min(len(unique2), half_n)
        remaining1 = half_n - max_unique1
        remaining2 = half_n - max_unique2
        
        common_to_use = min(len(common_elements), remaining1 + remaining2)
        
        return max_unique1 + max_unique2 + common_to_use