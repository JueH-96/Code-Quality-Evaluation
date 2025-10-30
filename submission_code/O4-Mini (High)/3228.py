from typing import List

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        # Compute distinct counts
        s1, s2 = set(nums1), set(nums2)
        d1, d2 = len(s1), len(s2)
        # Number of values present in both arrays
        common = len(s1 & s2)
        # Values only in nums1 or only in nums2
        only1 = d1 - common
        only2 = d2 - common
        
        half = len(nums1) // 2
        
        # How many "only" values we can include from each array
        take_only1 = min(only1, half)
        take_only2 = min(only2, half)
        
        # Remaining pick‐slots on each side for common values
        spare1 = half - take_only1
        spare2 = half - take_only2
        
        # We can include at most (spare1 + spare2) of the common values
        take_common = min(common, spare1 + spare2)
        
        # Total distinct = only1 + only2 + common (as assigned)
        return take_only1 + take_only2 + take_common