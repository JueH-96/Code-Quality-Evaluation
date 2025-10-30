from typing import List

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        # Number of slots we can keep in each array
        n2 = len(nums1) // 2
        
        s1 = set(nums1)
        s2 = set(nums2)
        
        # Distinct values exclusive to nums1, exclusive to nums2, and common
        only1 = s1 - s2
        only2 = s2 - s1
        common = s1 & s2
        
        C1 = len(only1)
        C2 = len(only2)
        C12 = len(common)
        
        # We can take at most n2 items from each side
        take1 = min(C1, n2)
        take2 = min(C2, n2)
        
        # Remaining capacity on each side for common items
        cap1 = n2 - take1
        cap2 = n2 - take2
        
        # We can assign a common value to either side; total we can place is cap1+cap2
        take_common = min(C12, cap1 + cap2)
        
        return take1 + take2 + take_common