from typing import List

class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        # Since every element in nums1 is increased by the same integer x
        # to form nums2, the total increase is len(nums1) * x.
        # Thus, x = (sum(nums2) - sum(nums1)) // len(nums1).
        return (sum(nums2) - sum(nums1)) // len(nums1)