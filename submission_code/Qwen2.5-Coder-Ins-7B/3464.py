class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        for i in range(1, n):
            dp[i] = max(dp[i - 1] + nums[i] * (1 if i % 2 == 0 else -1), nums[i] * (1 if i % 2 == 0 else -1))
        return dp[-1]