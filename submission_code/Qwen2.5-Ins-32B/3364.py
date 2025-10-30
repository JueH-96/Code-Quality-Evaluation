from typing import List

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n, m = len(nums), len(andValues)
        dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                and_val = 0
                for k in range(i - 1, -1, -1):
                    and_val &= nums[k]
                    if and_val == andValues[j - 1]:
                        dp[i][j] = min(dp[i][j], dp[k][j - 1] + nums[i - 1])
        
        return dp[n][m] if dp[n][m] != float('inf') else -1