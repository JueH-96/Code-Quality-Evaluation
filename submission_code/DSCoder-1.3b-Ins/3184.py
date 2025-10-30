class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = max(dp[i-1], nums[i-1] + max(dp[j] for j in range(i-1) if nums[j] < nums[i-1]))
        return dp[-1]