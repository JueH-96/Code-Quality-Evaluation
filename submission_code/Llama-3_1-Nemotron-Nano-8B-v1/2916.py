from typing import List

class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        # Initialize DP table where dp[i][k] means first i elements split into k parts
        dp = [[False] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = True  # Base case: 0 elements can be split into 0 parts
        
        for i in range(1, n + 1):
            for k in range(1, min(i, n) + 1):
                # Check all possible j to split into k parts
                for j in range(0, i):
                    if dp[j][k - 1]:
                        subarray_length = i - j
                        subarray_sum = prefix_sum[i] - prefix_sum[j]
                        if subarray_length == 1 or subarray_sum >= m:
                            dp[i][k] = True
                            break  # No need to check further once found
        return dp[n][n]