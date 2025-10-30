class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        # dp1[i] = length of longest non-decreasing subarray ending at i using nums1[i]
        # dp2[i] = length of longest non-decreasing subarray ending at i using nums2[i]
        dp1 = [1] * n
        dp2 = [1] * n
        
        max_length = 1
        
        for i in range(1, n):
            # If we choose nums1[i]
            if nums1[i] >= nums1[i-1]:
                dp1[i] = max(dp1[i], dp1[i-1] + 1)
            if nums1[i] >= nums2[i-1]:
                dp1[i] = max(dp1[i], dp2[i-1] + 1)
            
            # If we choose nums2[i]
            if nums2[i] >= nums1[i-1]:
                dp2[i] = max(dp2[i], dp1[i-1] + 1)
            if nums2[i] >= nums2[i-1]:
                dp2[i] = max(dp2[i], dp2[i-1] + 1)
            
            max_length = max(max_length, dp1[i], dp2[i])
        
        return max_length