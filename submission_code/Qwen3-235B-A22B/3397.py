from typing import List

class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        sum_diff = sum(nums2) - sum(nums1)
        return sum_diff // len(nums1)