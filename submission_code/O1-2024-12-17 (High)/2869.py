class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        # dp0: length of non-decreasing subarray ending at i if we pick nums1[i]
        # dp1: length of non-decreasing subarray ending at i if we pick nums2[i]
        dp0 = 1
        dp1 = 1
        ans = 1  # Tracks the maximum length so far
        
        for i in range(1, n):
            newdp0 = 1
            newdp1 = 1

            # If choosing nums1[i] continues from picking nums1[i-1]
            if nums1[i] >= nums1[i-1]:
                newdp0 = max(newdp0, dp0 + 1)
            # If choosing nums1[i] continues from picking nums2[i-1]
            if nums1[i] >= nums2[i-1]:
                newdp0 = max(newdp0, dp1 + 1)

            # If choosing nums2[i] continues from picking nums1[i-1]
            if nums2[i] >= nums1[i-1]:
                newdp1 = max(newdp1, dp0 + 1)
            # If choosing nums2[i] continues from picking nums2[i-1]
            if nums2[i] >= nums2[i-1]:
                newdp1 = max(newdp1, dp1 + 1)

            dp0, dp1 = newdp0, newdp1
            ans = max(ans, dp0, dp1)

        return ans