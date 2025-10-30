class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        for i in range(1, n):
            dp[i] = dp[i - 1] + nums[i] * (-1) ** i - i
        max_cost = max(0, dp[-1])
        max_dp = [dp[0]]
        for i in range(1, n):
            max_dp.append(max(max_dp[-1], dp[i]))
        for i in range(1, n):
            max_cost = max(max_cost, max_dp[i - 1] + dp[-1] - dp[i - 1] + nums[i] * (-1) ** i - i)
        return max_cost