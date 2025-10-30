from typing import List

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        dp[n - 1] = nums[n - 1]
        
        for i in range(n - 2, -1, -1):
            max_score = 0
            for j in range(i + 1, n):
                max_score = max(max_score, (j - i) * nums[i] + dp[j])
            dp[i] = max_score
        
        return dp[0]