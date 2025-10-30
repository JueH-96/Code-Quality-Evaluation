from typing import List

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [-1] * n
        dp[0] = 0  # Starting at index 0 with 0 jumps
        
        for i in range(1, n):
            for j in range(i):
                diff = nums[i] - nums[j]
                if -target <= diff <= target:
                    if dp[j] != -1:
                        if dp[i] < dp[j] + 1:
                            dp[i] = dp[j] + 1
        
        return dp[-1] if dp[-1] != -1 else -1