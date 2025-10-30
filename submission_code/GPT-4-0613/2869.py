class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        dp = [[0]*2 for _ in range(n+1)]
        for i in range(n-1, -1, -1):
            if nums1[i] <= nums1[i+1] and nums2[i] <= nums2[i+1]:
                if nums1[i] <= nums2[i+1] and nums2[i] <= nums1[i+1]:
                    dp[i][0] = dp[i][1] = max(dp[i+1][0], dp[i+1][1]) + 1
                else:
                    dp[i][0] = dp[i+1][0] + 1
                    dp[i][1] = dp[i+1][1] + 1
            else:
                if nums1[i] <= nums1[i+1]:
                    dp[i][0] = dp[i+1][0] + 1
                if nums2[i] <= nums2[i+1]:
                    dp[i][1] = dp[i+1][1] + 1
                if nums1[i] <= nums2[i+1]:
                    dp[i][0] = max(dp[i][0], dp[i+1][1] + 1)
                if nums2[i] <= nums1[i+1]:
                    dp[i][1] = max(dp[i][1], dp[i+1][0] + 1)
        return max(dp[0][0], dp[0][1])