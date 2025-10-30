class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i] * (-1) ** i

        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = max(dp[i - 1], prefix_sum[i] - min(prefix_sum[:i]))

        return dp[n]