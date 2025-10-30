from typing import List

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[-float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for x in range(1, k + 1):
                dp[i][x] = max(dp[i][x], dp[i - 1][x])
                dp[i][x] = max(dp[i][x], dp[i - 1][x - 1] + nums[i - 1] * (x if x % 2 else -x))
        
        return max(dp[i][k] for i in range(n + 1))