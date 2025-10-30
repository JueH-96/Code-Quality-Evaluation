class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        diff = [nums2[i] - nums1[i] for i in range(n)]
        total_diff = sum(diff)
        if total_diff <= x:
            return n
        dp = [0] + [-float('inf')] * x
        for i in range(n):
            for j in range(x, diff[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - diff[i]] + 1)
        return dp[x] if dp[x] >= 0 else -1