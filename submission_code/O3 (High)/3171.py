from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        # current sums
        s1, s2 = sum(nums1), sum(nums2)
        # number of positions we are allowed to change (the zeros)
        z1, z2 = nums1.count(0), nums2.count(0)
        
        # ------------------------------------------------------------------
        # 1) neither array contains zeros -> nothing can be changed
        # ------------------------------------------------------------------
        if z1 == 0 and z2 == 0:
            return s1 if s1 == s2 else -1
        
        # ------------------------------------------------------------------
        # 2) only nums1 is fixed (z1 == 0)
        #    we must be able to raise nums2 up to s1,
        #    whose minimum attainable value is   s2 + z2  (all zeros -> 1)
        # ------------------------------------------------------------------
        if z1 == 0:
            return s1 if s1 >= s2 + z2 else -1
        
        # ------------------------------------------------------------------
        # 3) only nums2 is fixed (z2 == 0) – symmetric case
        # ------------------------------------------------------------------
        if z2 == 0:
            return s2 if s2 >= s1 + z1 else -1
        
        # ------------------------------------------------------------------
        # 4) both arrays have at least one zero
        #    we can always increase either side, so the smallest common
        #    sum is simply the larger of the two minimal possible sums
        # ------------------------------------------------------------------
        return max(s1 + z1, s2 + z2)