from typing import List

class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        sorted_nums1 = sorted(nums1)
        sorted_nums2 = sorted(nums2)
        return sorted_nums2[0] - sorted_nums1[0]