class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 0:
            return 0
        # Initialize dp arrays
        # dp1[i] represents the length of the longest non-decreasing subarray ending at index i when nums3[i] is nums1[i]
        # dp2[i] represents the length of the longest non-decreasing subarray ending at index i when nums3[i] is nums2[i]
        dp1 = [1] * n
        dp2 = [1] * n
        max_len = 1
        for i in range(1, n):
            # For dp1[i], we can choose nums1[i] and see if it can extend the subarray from dp1[i-1] or dp2[i-1]
            if nums1[i] >= nums1[i-1]:
                dp1[i] = max(dp1[i], dp1[i-1] + 1)
            if nums1[i] >= nums2[i-1]:
                dp1[i] = max(dp1[i], dp2[i-1] + 1)
            # For dp2[i], we can choose nums2[i] and see if it can extend the subarray from dp1[i-1] or dp2[i-1]
            if nums2[i] >= nums1[i-1]:
                dp2[i] = max(dp2[i], dp1[i-1] + 1)
            if nums2[i] >= nums2[i-1]:
                dp2[i] = max(dp2[i], dp2[i-1] + 1)
            # Update the maximum length
            max_len = max(max_len, dp1[i], dp2[i])
        return max_len