class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        for i in range(n - 1):
            for j in range(i + 1, n):
                dp[j] = max(dp[j], dp[i] + (j - i) * nums[i])
        return dp[n - 1]