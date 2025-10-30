from typing import List
import math

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        
        # Initialize DP table with infinity
        dp = [[math.inf] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                current_and = 0
                for k in range(i, 0, -1):
                    current_and &= nums[k - 1]
                    if current_and == andValues[j - 1]:
                        dp[i][j] = min(dp[i][j], dp[k - 1][j - 1] + nums[i - 1])
        
        return dp[n][m] if dp[n][m] != math.inf else -1