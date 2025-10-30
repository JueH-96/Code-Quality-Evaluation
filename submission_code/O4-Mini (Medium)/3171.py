from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        # Compute sum of non-zero elements and count zeros in each array
        s1 = sum(nums1)
        k1 = nums1.count(0)
        s2 = sum(nums2)
        k2 = nums2.count(0)
        
        # If neither array has zeros, sums must already match
        if k1 == 0 and k2 == 0:
            return s1 if s1 == s2 else -1
        
        # If nums1 has no zeros, its sum is fixed; we need s1 = s2 + y with y >= k2
        if k1 == 0:
            # y = s1 - s2 must be >= k2
            return s1 if s1 >= s2 + k2 else -1
        
        # If nums2 has no zeros, its sum is fixed; we need s2 = s1 + x with x >= k1
        if k2 == 0:
            # x = s2 - s1 must be >= k1
            return s2 if s2 >= s1 + k1 else -1
        
        # Both have zeros: we can choose replacements so that
        # final sum S >= s1 + k1 and S >= s2 + k2, minimal S is the maximum of those
        return max(s1 + k1, s2 + k2)