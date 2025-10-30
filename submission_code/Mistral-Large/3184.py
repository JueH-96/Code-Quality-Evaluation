from typing import List

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]

        dp = nums.copy()

        for i in range(1, n):
            for j in range(i):
                if nums[i] - nums[j] >= i - j:
                    dp[i] = max(dp[i], dp[j] + nums[i])

        return max(dp)