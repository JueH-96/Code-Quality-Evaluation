from typing import List

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        set1 = set(nums1)
        set2 = set(nums2)
        
        common = set1.intersection(set2)
        unique1 = set1 - common
        unique2 = set2 - common
        
        take_from_common = min(len(common), n // 2)
        take_from_unique1 = min(len(unique1), n // 2 - take_from_common)
        take_from_unique2 = min(len(unique2), n // 2 - take_from_common)
        
        return take_from_common + take_from_unique1 + take_from_unique2