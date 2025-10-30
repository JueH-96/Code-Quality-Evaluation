class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        nums = sorted(zip(nums1, nums2), key=lambda x: x[1])
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            a, b = nums[i - 1]
            for j in range(1, i + 1):
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] + a + b * j)
        s1 = sum(nums1)
        s2 = sum(nums2)
        for t in range(n + 1):
            if s1 + s2 * t - dp[n][t] <= x:
                return t
        return -1