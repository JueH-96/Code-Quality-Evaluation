from typing import List

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [float('-inf')] * n
        dp[0] = 0  # starting point requires 0 jumps
        
        for i in range(1, n):
            for j in range(i):
                diff = nums[i] - nums[j]
                if -target <= diff <= target:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
        
        return dp[-1] if dp[-1] != float('-inf') else -1