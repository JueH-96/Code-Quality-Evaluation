from typing import List

class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        # dp1[i] is the maximum non-decreasing subarray ending at i if we choose nums1[i]
        # dp2[i] is the maximum non-decreasing subarray ending at i if we choose nums2[i]
        dp1 = [0] * n
        dp2 = [0] * n
        
        dp1[0] = 1
        dp2[0] = 1
        res = 1
        for i in range(1, n):
            best1 = 1  # if we decide to start afresh at i
            best2 = 1  # similar for nums2[i]
            
            # Option for dp1: choose nums1[i]
            # Check if we can extend from previous dp1 (if nums1[i-1] <= nums1[i])
            if nums1[i-1] <= nums1[i]:
                best1 = max(best1, dp1[i-1] + 1)
            # Check if we can extend from previous dp2 (if nums2[i-1] <= nums1[i])
            if nums2[i-1] <= nums1[i]:
                best1 = max(best1, dp2[i-1] + 1)
            
            # Option for dp2: choose nums2[i]
            # Check from dp1
            if nums1[i-1] <= nums2[i]:
                best2 = max(best2, dp1[i-1] + 1)
            # Check from dp2
            if nums2[i-1] <= nums2[i]:
                best2 = max(best2, dp2[i-1] + 1)
            
            dp1[i] = best1
            dp2[i] = best2
            res = max(res, best1, best2)
            
        return res