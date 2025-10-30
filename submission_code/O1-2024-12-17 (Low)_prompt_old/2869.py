class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        # dp1[i] = length of longest non-decreasing subarray ending at i if we pick nums1[i]
        # dp2[i] = length of longest non-decreasing subarray ending at i if we pick nums2[i]
        dp1 = [1] * n
        dp2 = [1] * n
        
        ans = 1  # at least one element can form a non-decreasing subarray
        
        for i in range(1, n):
            # If we pick nums1[i], extend from previous if possible
            # 1) if previous pick was nums1[i-1] and nums1[i] >= nums1[i-1], extend dp1[i-1]
            # 2) if previous pick was nums2[i-1] and nums1[i] >= nums2[i-1], extend dp2[i-1]
            dp1[i] = max(
                (dp1[i-1] + 1) if nums1[i] >= nums1[i-1] else 1,
                (dp2[i-1] + 1) if nums1[i] >= nums2[i-1] else 1
            )
            
            # If we pick nums2[i], similarly check previous picks
            dp2[i] = max(
                (dp1[i-1] + 1) if nums2[i] >= nums1[i-1] else 1,
                (dp2[i-1] + 1) if nums2[i] >= nums2[i-1] else 1
            )
            
            ans = max(ans, dp1[i], dp2[i])
        
        return ans