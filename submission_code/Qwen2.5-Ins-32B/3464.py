from typing import List

class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            dp[i] = max(nums[i] - dp[i + 1], dp[i + 1] + nums[i])
        return dp[0]