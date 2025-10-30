from typing import List

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n
        prefix_sum = [0] * (n + 1)
        
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        for i in range(1, n):
            for j in range(i, -1, -1):
                if prefix_sum[i + 1] - prefix_sum[j] >= (prefix_sum[j] - prefix_sum[max(0, j - dp[j] + 1)] if j > 0 else 0):
                    dp[i] = max(dp[i], dp[j - 1] + 1 if j > 0 else 1)
                    break
        
        return dp[-1]