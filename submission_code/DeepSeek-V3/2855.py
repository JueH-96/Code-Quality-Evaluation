from typing import List

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        if n == 0:
            return -1
        dp = [-1] * n
        dp[0] = 0
        for i in range(n):
            if dp[i] == -1:
                continue
            for j in range(i+1, n):
                if -target <= nums[j] - nums[i] <= target:
                    if dp[j] < dp[i] + 1:
                        dp[j] = dp[i] + 1
        return dp[-1] if dp[-1] != -1 else -1