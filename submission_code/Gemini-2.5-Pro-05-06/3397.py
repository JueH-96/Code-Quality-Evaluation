from typing import List

class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # The transformation nums1[i] -> nums1[i] + x preserves the relative order
        # of elements. Specifically, the minimum element of the transformed nums1
        # will be min(nums1) + x.
        #
        # Since the transformed nums1 is equal to nums2 (as multisets),
        # their minimum elements must be equal.
        # So, min(nums1) + x = min(nums2).
        #
        # This gives x = min(nums2) - min(nums1).
        
        # Calculate the minimum element of nums1.
        # nums1 is guaranteed to be non-empty (length >= 1).
        min_val_nums1 = min(nums1)
        
        # Calculate the minimum element of nums2.
        # nums2 is guaranteed to be non-empty.
        min_val_nums2 = min(nums2)
        
        # The integer x is the difference.
        # Since all elements in nums1 and nums2 are integers, min_val_nums1 and
        # min_val_nums2 will be integers. Thus, x will also be an integer.
        return min_val_nums2 - min_val_nums1