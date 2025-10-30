from typing import List

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0

        dp = [-float('inf')] * n
        dp[0] = 0

        for i in range(1, n):
            for j in range(i):
                score = dp[j] + (i - j) * nums[j]
                dp[i] = max(dp[i], score)

        return dp[n - 1]