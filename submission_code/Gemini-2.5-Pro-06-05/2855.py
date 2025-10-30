from typing import List

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        
        # dp[i] stores the maximum number of jumps to reach index i.
        # Initialize with -1 to signify an unreachable state.
        dp = [-1] * n
        
        # We start at index 0 with 0 jumps.
        dp[0] = 0
        
        # Build up the dp table from left to right.
        for i in range(1, n):
            # Check all possible previous indices j to jump from.
            for j in range(i):
                # A jump is possible if j is reachable and the jump condition is met.
                if dp[j] != -1 and abs(nums[i] - nums[j]) <= target:
                    # We found a valid path to i. The number of jumps is 1 + jumps to j.
                    # We want the maximum number of jumps, so we update dp[i].
                    dp[i] = max(dp[i], dp[j] + 1)
                    
        # The result is the value for the last index.
        # If it's still -1, the last index is unreachable.
        return dp[n - 1]