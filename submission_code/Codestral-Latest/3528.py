class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = 0

        for i in range(1, n):
            max_score = 0
            for j in range(i):
                score = dp[j] + (i - j) * nums[j]
                if score > max_score:
                    max_score = score
            dp[i] = max_score

        return dp[n - 1]