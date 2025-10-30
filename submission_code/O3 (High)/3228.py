from typing import List

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        k = n // 2                     # elements that have to remain in each array
        
        set1, set2 = set(nums1), set(nums2)
        
        only1 = set1 - set2            # values that exist only in nums1
        only2 = set2 - set1            # values that exist only in nums2
        common = set1 & set2           # values that exist in both arrays
        
        # take as many exclusive values as possible from each array
        keep1 = min(len(only1), k)     # distinct values we keep from nums1 only
        keep2 = min(len(only2), k)     # distinct values we keep from nums2 only
        
        # remaining capacity in each array can be filled with common values
        rem1 = k - keep1               # still free “slots” in nums1
        rem2 = k - keep2               # still free “slots” in nums2
        
        common_kept = min(len(common), rem1 + rem2)
        
        return keep1 + keep2 + common_kept