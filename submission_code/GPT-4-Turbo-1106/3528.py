class Solution:
    def findMaximumScore(self, nums: list[int]) -> int:
        n = len(nums)
        dp = [0] * n
        max_score = 0
        
        for i in range(n - 2, -1, -1):
            dp[i] = nums[i] * (n - 1 - i) + dp[n - 1]
            for j in range(n - 1, i, -1):
                dp[i] = max(dp[i], nums[i] * (j - i) + dp[j])
            max_score = max(max_score, dp[i])
        
        return max_score