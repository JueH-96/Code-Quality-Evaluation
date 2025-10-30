from typing import List

class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        # dp0 = length ending at i-1 if we chose nums1[i-1]
        # dp1 = length ending at i-1 if we chose nums2[i-1]
        dp0 = dp1 = 1
        res = 1
        
        for i in range(1, n):
            x1, x2 = nums1[i], nums2[i]
            y1, y2 = nums1[i-1], nums2[i-1]
            
            new_dp0 = 1
            # if we pick nums1[i] = x1
            if y1 <= x1:
                new_dp0 = max(new_dp0, dp0 + 1)
            if y2 <= x1:
                new_dp0 = max(new_dp0, dp1 + 1)
            
            new_dp1 = 1
            # if we pick nums2[i] = x2
            if y1 <= x2:
                new_dp1 = max(new_dp1, dp0 + 1)
            if y2 <= x2:
                new_dp1 = max(new_dp1, dp1 + 1)
            
            dp0, dp1 = new_dp0, new_dp1
            res = max(res, dp0, dp1)
        
        return res