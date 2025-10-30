from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(x for x in nums1 if x != 0)
        sum2 = sum(x for x in nums2 if x != 0)

        diff = sum2 - sum1

        if diff == 0:
            if nums1.count(0) > 0 or nums2.count(0) > 0:
                return sum1
            else:
                return -1

        if diff > 0:
            if nums1.count(0) == 0:
                return -1
            return sum2 + (nums1.count(0) if nums1.count(0) > 0 else 0)
        else:
            if nums2.count(0) == 0:
                return -1
            return sum1 + (nums2.count(0) if nums2.count(0) > 0 else 0)