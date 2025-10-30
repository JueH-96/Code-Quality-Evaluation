from typing import List

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n, m = len(nums), len(andValues)
        if n < m:
            return -1

        dp = [float('inf')] * m
        dp[-1] = nums[-1] if nums[-1] & andValues[-1] == andValues[-1] else float('inf')
        for i in range(n - 2, -1, -1):
            current_and = nums[i]
            for j in range(m - 1, -1, -1):
                if current_and & andValues[j] == andValues[j]:
                    if j == m - 1:
                        dp[j] = min(dp[j], nums[i])
                    else:
                        dp[j] = min(dp[j], nums[i] + dp[j + 1])
                current_and &= nums[i + 1] if i + 1 < n else 0

        return dp[0] if dp[0] != float('inf') else -1