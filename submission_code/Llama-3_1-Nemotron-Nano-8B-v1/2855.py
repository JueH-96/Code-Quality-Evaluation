from typing import List

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        if n == 0:
            return -1
        dp = [-float('inf')] * n
        dp[0] = 0  # Starting point
        
        for j in range(1, n):
            for i in range(j):
                if abs(nums[j] - nums[i]) <= target:
                    if dp[i] + 1 > dp[j]:
                        dp[j] = dp[i] + 1
        
        return dp[-1] if dp[-1] != -float('inf') else -1