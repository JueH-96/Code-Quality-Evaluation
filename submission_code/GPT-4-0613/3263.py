class Solution:
    def minimumCost(self, nums):
        n = len(nums)
        dp = [[float('inf')] * n for _ in range(3)]
        dp[0][0], dp[1][1], dp[2][2] = nums[0], nums[0] + nums[1], nums[0] + nums[1] + nums[2]
        for i in range(1, n):
            dp[0][i] = min(dp[0][i-1], nums[i])
        for i in range(2, n):
            dp[1][i] = min(dp[1][i-1], dp[0][i-1] + nums[i])
        for i in range(3, n):
            dp[2][i] = min(dp[2][i-1], dp[1][i-1] + nums[i])
        return dp[2][-1]