from typing import List

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        dp = [0] * n

        dp[0] = nums[0]

        for i in range(1, n):
            dp[i] = dp[i - 1] + nums[i]
            if (nums[i] % 2) != (nums[i - 1] % 2):
                for j in range(i - 1):
                    dp[i] = max(dp[i], dp[j] + nums[i] - x)

        return max(dp)