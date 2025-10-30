class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        dp[n - 1] = 0
        
        for i in range(n - 2, -1, -1):
            max_score = (n - 1 - i) * nums[i]
            for j in range(i + 1, n - 1):
                score = (j - i) * nums[i] + dp[j]
                max_score = max(max_score, score)
            dp[i] = max_score
        
        return dp[0]