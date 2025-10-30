class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        max_score = nums[0]

        for i in range(1, n):
            dp[i] = max(dp[i-1], max_score + nums[i])
            max_score = max(max_score, dp[i-1] - nums[i-1] + nums[i])

        return dp[-1]