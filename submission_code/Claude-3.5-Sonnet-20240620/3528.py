class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        dp[-1] = 0  # Base case: last index has no score to add

        for i in range(n - 2, -1, -1):
            max_score = 0
            for j in range(i + 1, n):
                score = (j - i) * nums[i] + dp[j]
                max_score = max(max_score, score)
            dp[i] = max_score

        return dp[0]