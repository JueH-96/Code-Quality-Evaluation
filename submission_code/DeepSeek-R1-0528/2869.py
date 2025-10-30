class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        dp1, dp2 = 1, 1
        max_len = 1
        
        for i in range(1, n):
            new_dp1 = 1
            new_dp2 = 1
            
            if nums1[i-1] <= nums1[i]:
                new_dp1 = max(new_dp1, dp1 + 1)
            if nums2[i-1] <= nums1[i]:
                new_dp1 = max(new_dp1, dp2 + 1)
            
            if nums1[i-1] <= nums2[i]:
                new_dp2 = max(new_dp2, dp1 + 1)
            if nums2[i-1] <= nums2[i]:
                new_dp2 = max(new_dp2, dp2 + 1)
            
            dp1, dp2 = new_dp1, new_dp2
            max_len = max(max_len, dp1, dp2)
        
        return max_len