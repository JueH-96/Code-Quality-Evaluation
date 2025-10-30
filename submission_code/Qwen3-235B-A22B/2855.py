from typing import List

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [float('-inf')] * n
        dp[0] = 0  # Starting at index 0 with 0 jumps
        
        for i in range(n):
            for j in range(i):
                if abs(nums[i] - nums[j]) <= target:
                    if dp[j] != float('-inf'):
                        dp[i] = max(dp[i], dp[j] + 1)
        
        return dp[-1] if dp[-1] != float('-inf') else -1