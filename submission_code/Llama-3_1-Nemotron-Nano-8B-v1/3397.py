from typing import List

class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        nums1_sorted = sorted(nums1)
        nums2_sorted = sorted(nums2)
        return nums2_sorted[0] - nums1_sorted[0]