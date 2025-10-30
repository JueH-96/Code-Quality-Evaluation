from typing import List

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [-float('inf')] * n
        dp[0] = 0  # Starting point, 0 jumps
        
        for j in range(1, n):
            for i in range(j):
                diff = nums[j] - nums[i]
                if -target <= diff <= target:
                    if dp[i] != -float('inf'):
                        dp[j] = max(dp[j], dp[i] + 1)
        
        return dp[-1] if dp[-1] != -float('inf') else -1