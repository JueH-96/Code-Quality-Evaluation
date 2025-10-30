from typing import List

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [-1] * n
        dp[n - 1] = 0  # Base case: 0 jumps needed to reach the last index from itself
        
        for i in range(n - 2, -1, -1):
            for j in range(i + 1, n):
                if abs(nums[j] - nums[i]) <= target:
                    if dp[j] != -1:
                        dp[i] = max(dp[i], dp[j] + 1)
        
        return dp[0]