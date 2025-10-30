from typing import List

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        # dp[i] will hold the maximum number of jumps to reach index i,
        # or -1 if unreachable.
        dp = [-1] * n
        dp[0] = 0
        
        for i in range(1, n):
            best = -1
            for j in range(i):
                if dp[j] >= 0 and abs(nums[i] - nums[j]) <= target:
                    # we can jump from j to i
                    best = max(best, dp[j] + 1)
            dp[i] = best
        
        return dp[n-1] if dp[n-1] >= 0 else -1