class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 0:
            return 0
        # Initialize dp arrays
        dp1 = [1] * n  # dp1[i]: max length ending at i, choosing nums1[i]
        dp2 = [1] * n  # dp2[i]: max length ending at i, choosing nums2[i]
        
        for i in range(1, n):
            # If we choose nums1[i], we can choose nums1[i-1] or nums2[i-1]
            if nums1[i] >= nums1[i-1]:
                dp1[i] = max(dp1[i], dp1[i-1] + 1)
            if nums1[i] >= nums2[i-1]:
                dp1[i] = max(dp1[i], dp2[i-1] + 1)
            # If we choose nums2[i], we can choose nums1[i-1] or nums2[i-1]
            if nums2[i] >= nums1[i-1]:
                dp2[i] = max(dp2[i], dp1[i-1] + 1)
            if nums2[i] >= nums2[i-1]:
                dp2[i] = max(dp2[i], dp2[i-1] + 1)
        
        # The answer is the maximum value in dp1 and dp2
        return max(max(dp1), max(dp2))