from typing import List

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = 0
        
        for i in range(1, n):
            for j in range(i):
                dp[i] = max(dp[i], dp[j] + (i - j) * nums[j])
        
        return dp[-1]