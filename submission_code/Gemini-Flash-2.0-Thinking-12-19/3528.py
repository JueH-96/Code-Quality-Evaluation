class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        dp = [0] * n
        for i in range(1, n):
            max_score = 0
            for j in range(i):
                max_score = max(max_score, dp[j] + (i - j) * nums[j])
            dp[i] = max_score
        return dp[n - 1]