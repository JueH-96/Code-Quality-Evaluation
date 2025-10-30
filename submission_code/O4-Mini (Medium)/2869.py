from typing import List

class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        # dp1: max length of non-decreasing subarray ending at i if we pick nums1[i]
        # dp2: same if we pick nums2[i]
        dp1 = dp2 = 1
        ans = 1
        
        for i in range(1, n):
            new_dp1 = 1
            new_dp2 = 1
            
            # If we choose nums1[i]:
            if nums1[i] >= nums1[i-1]:
                new_dp1 = max(new_dp1, dp1 + 1)
            if nums1[i] >= nums2[i-1]:
                new_dp1 = max(new_dp1, dp2 + 1)
            
            # If we choose nums2[i]:
            if nums2[i] >= nums1[i-1]:
                new_dp2 = max(new_dp2, dp1 + 1)
            if nums2[i] >= nums2[i-1]:
                new_dp2 = max(new_dp2, dp2 + 1)
            
            dp1, dp2 = new_dp1, new_dp2
            ans = max(ans, dp1, dp2)
        
        return ans