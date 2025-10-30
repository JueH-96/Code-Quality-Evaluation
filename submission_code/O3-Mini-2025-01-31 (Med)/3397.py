from typing import List

class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Since each element in nums1 has x added to it to obtain nums2,
        # we have nums1[i] + x = nums2[i] for every i.
        # Thus, x = nums2[i] - nums1[i], which is constant across all indices.
        # We can directly return the difference for the first element.
        return nums2[0] - nums1[0]