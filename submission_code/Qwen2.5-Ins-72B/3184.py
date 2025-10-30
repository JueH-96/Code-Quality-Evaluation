from typing import List
import bisect

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        sorted_dp = []
        
        for i in range(n):
            dp[i] = nums[i]
            idx = bisect.bisect_left(sorted_dp, dp[i] - i)
            if idx > 0:
                dp[i] += sorted_dp[idx - 1] + i
            if idx == len(sorted_dp):
                sorted_dp.append(dp[i] - i)
            else:
                sorted_dp[idx] = dp[i] - i
        
        return max(dp)