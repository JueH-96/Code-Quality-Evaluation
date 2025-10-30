from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        
        # Precompute the prefix sums
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        # dp[i][j] means the maximum sum we can get with j subarrays using the first i elements
        dp = [[-float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        # Iterate over the array
        for i in range(1, n + 1):
            for j in range(k + 1):
                # Carry forward the previous maximum sum
                dp[i][j] = dp[i - 1][j]
                
                # Try to form a new subarray ending at i-1
                if j > 0 and i >= m:
                    for l in range(m, i + 1):
                        dp[i][j] = max(dp[i][j], dp[i - l][j - 1] + prefix_sum[i] - prefix_sum[i - l])
        
        return dp[n][k]