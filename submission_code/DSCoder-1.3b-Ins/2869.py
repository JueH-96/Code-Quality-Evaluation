class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        dp = [[0]*n for _ in range(n)]
        dp2 = [[0]*n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
            dp2[i][i] = 1
            if i+1 < n:
                if nums1[i] <= nums1[i+1]:
                    dp[i][i+1] = 2
                else:
                    dp[i][i+1] = 1
                if nums2[i] <= nums2[i+1]:
                    dp2[i][i+1] = 2
                else:
                    dp2[i][i+1] = 1
        for length in range(2, n):
            for i in range(n-length):
                j = i + length
                if nums1[i] <= nums1[j] and nums2[i] <= nums2[j]:
                    dp[i][j] = max(dp[i][j], dp[i+1][j] + 1)
                if nums1[i] <= nums2[j] and nums2[i] <= nums2[j]:
                    dp2[i][j] = max(dp2[i][j], dp2[i+1][j] + 1)
                if nums1[i] <= nums1[j] and nums2[j] <= nums2[i]:
                    dp[i][j] = max(dp[i][j], dp[i+1][j] + 1)
                if nums2[i] <= nums2[j] and nums1[j] <= nums1[i]:
                    dp2[i][j] = max(dp2[i][j], dp2[i+1][j] + 1)
        return max(max(dp[i][j] for i in range(n)), max(dp2[i][j] for i in range(n)))