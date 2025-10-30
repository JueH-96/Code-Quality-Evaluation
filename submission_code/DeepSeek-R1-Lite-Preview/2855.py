from typing import List

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [-1] * n
        dp[0] = 0  # Starting point requires 0 jumps
        for i in range(1, n):
            for j in range(i):
                diff = nums[i] - nums[j]
                if -target <= diff <= target and dp[j] != -1:
                    dp[i] = max(dp[i], dp[j] + 1)
        return dp[-1]