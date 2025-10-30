class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        dp = [0] * n
        dp[0] = 0
        for i in range(1, n):
            max_score = 0
            for j in range(i):
                current_score = dp[j] + (i - j) * nums[j]
                if current_score > max_score:
                    max_score = current_score
            dp[i] = max_score
        return dp[-1]