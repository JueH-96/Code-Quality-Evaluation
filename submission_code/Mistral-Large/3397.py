from typing import List

class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Since nums1 and nums2 are of equal length and the problem guarantees
        # that there exists an integer x such that nums1[i] + x = nums2[i] for all i,
        # we can simply calculate x by taking the difference between the first elements
        # of nums2 and nums1.
        return nums2[0] - nums1[0]