from typing import List

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        dp = [-1] * (target + 1)
        dp[0] = 0

        for num in nums:
            if num <= target and dp[target - num] != -1:
                dp[target] = max(dp[target], dp[target - num] + 1)

        return dp[target]