from typing import List

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        if n == 0:
            return -1
        # Initialize dp array where dp[i] represents the maximum number of jumps to reach index i
        dp = [-1] * n
        dp[0] = 0  # Starting point, no jumps needed
        
        for i in range(n):
            if dp[i] == -1:
                continue  # Cannot reach this index, skip
            for j in range(i+1, n):
                if -target <= nums[j] - nums[i] <= target:
                    if dp[j] < dp[i] + 1:
                        dp[j] = dp[i] + 1
        
        return dp[-1]