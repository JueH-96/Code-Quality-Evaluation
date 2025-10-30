class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            dp[i] = max(dp[i + 1], nums[i] + (dp[i + 2] if i + 2 < n else 0))
        return dp[0]