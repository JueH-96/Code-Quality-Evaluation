class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        if n > 1:
            dp[1] = max(nums[0] + nums[1], nums[0] - nums[1])
        for i in range(2, n):
            dp[i] = max(dp[i - 1] + nums[i] * (1 if (i % 2 == 0) else -1), dp[i - 2] + nums[i] * (1 if (i % 2 == 0) else -1) + nums[i-1] * (1 if (i % 2 != 0) else -1))

        return dp[n - 1]