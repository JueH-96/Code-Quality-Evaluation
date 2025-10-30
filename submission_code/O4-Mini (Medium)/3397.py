from typing import List

class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Since each element in nums1 is increased by the same integer x to get nums2,
        # the total sum difference between nums2 and nums1 is n * x, where n = len(nums1).
        # Therefore, x = (sum(nums2) - sum(nums1)) / n.
        return (sum(nums2) - sum(nums1)) // len(nums1)