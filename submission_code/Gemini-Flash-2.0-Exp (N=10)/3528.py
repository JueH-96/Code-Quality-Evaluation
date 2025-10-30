class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return 0
        
        dp = [0] * n
        
        for i in range(n - 2, -1, -1):
            max_score = 0
            for j in range(i + 1, n):
                score = (j - i) * nums[i]
                if j == n - 1:
                    max_score = max(max_score, score)
                else:
                    max_score = max(max_score, score + dp[j])
            dp[i] = max_score
        
        return dp[0]