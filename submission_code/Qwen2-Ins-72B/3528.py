from typing import List

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = 0
        max_score = 0
        for i in range(1, n):
            for j in range(i):
                score = (i - j) * nums[j]
                if dp[j] + score > max_score:
                    max_score = dp[j] + score
            dp[i] = max_score
        return max_score