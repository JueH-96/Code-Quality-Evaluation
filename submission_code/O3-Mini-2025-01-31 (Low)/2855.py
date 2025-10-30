from typing import List

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        # dp[i] means maximum number of jumps from index 0 to i.
        # start point dp[0] = 0 jumps.
        dp = [-1] * n
        dp[0] = 0
        
        for i in range(1, n):
            # Check all previous indices j that can reach i.
            for j in range(i):
                if dp[j] != -1 and -target <= nums[i] - nums[j] <= target:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return dp[n-1]