class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [-(10**18)] * n
        for i in range(n):
            dp[i] = nums[i]
            for j in range(i):
                if nums[i] - nums[j] >= i - j:
                    dp[i] = max(dp[i], dp[j] + nums[i])
        return max(dp)