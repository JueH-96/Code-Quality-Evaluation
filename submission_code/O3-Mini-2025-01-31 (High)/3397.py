from typing import List

class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # The idea is that when we add x to each element of nums1,
        # the new sum becomes sum(nums1) + len(nums1) * x, which should equal sum(nums2).
        # Thus, x = (sum(nums2) - sum(nums1)) / len(nums1)
        n = len(nums1)
        diff = sum(nums2) - sum(nums1)
        return diff // n