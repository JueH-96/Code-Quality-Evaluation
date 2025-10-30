from typing import List

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n, m = len(nums), len(andValues)
        dp = [[float('inf')] * (1 << m) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for mask in range(1 << m):
                dp[i][mask] = dp[i - 1][mask]
                for j in range(m):
                    if mask & (1 << j) == 0:
                        and_val = andValues[j]
                        for k in range(i - 1, -1, -1):
                            and_val &= nums[k]
                            if and_val == andValues[j]:
                                dp[i][mask | (1 << j)] = min(dp[i][mask | (1 << j)], dp[k][mask] + nums[i - 1])
                            elif and_val < andValues[j]:
                                break
        
        result = min(dp[n][mask] for mask in range(1 << m) if mask == (1 << m) - 1)
        return result if result != float('inf') else -1