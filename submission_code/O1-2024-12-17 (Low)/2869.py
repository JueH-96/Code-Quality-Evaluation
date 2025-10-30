class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        # dp1[i] = length of the longest non-decreasing subarray ending at i if we pick nums1[i]
        # dp2[i] = length of the longest non-decreasing subarray ending at i if we pick nums2[i]
        # Recurrence:
        # dp1[i] = 1 + max(dp1[i-1] if nums1[i-1] <= nums1[i], dp2[i-1] if nums2[i-1] <= nums1[i])
        # dp2[i] = 1 + max(dp1[i-1] if nums1[i-1] <= nums2[i], dp2[i-1] if nums2[i-1] <= nums2[i])
        
        dp1, dp2 = 1, 1  # for i = 0
        max_length = 1
        
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
            
            # Update the dp values
            dp1, dp2 = new_dp1, new_dp2
            
            # Track the global maximum
            max_length = max(max_length, dp1, dp2)
        
        return max_length