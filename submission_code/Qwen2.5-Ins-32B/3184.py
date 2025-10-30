from typing import List
import bisect

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        max_sum = float('-inf')
        for i, num in enumerate(nums):
            idx = bisect.bisect_left(dp, num - i)
            if idx == 0:
                dp[0] = num - i
            else:
                dp[idx] = min(dp[idx], num - i)
            max_sum = max(max_sum, num + i + dp[idx])
        return max_sum