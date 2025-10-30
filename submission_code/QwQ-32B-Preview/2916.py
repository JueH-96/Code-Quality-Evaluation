from typing import List

class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        if n == 1:
            return True
        
        # Compute prefix sums
        prefix_sums = [0] * (n + 1)
        for i in range(n):
            prefix_sums[i + 1] = prefix_sums[i] + nums[i]
        
        # Initialize DP table
        dp = [[False] * n for _ in range(n)]
        
        # Base case: single element subarrays
        for i in range(n):
            dp[i][i] = True
        
        # Fill DP table for subarrays of length 2 to n
        for length in range(2, n + 1):
            for left in range(n - length + 1):
                right = left + length - 1
                subarray_sum = prefix_sums[right + 1] - prefix_sums[left]
                if subarray_sum >= m:
                    for k in range(left, right):
                        if dp[left][k] and dp[k + 1][right]:
                            dp[left][right] = True
                            break
        
        return dp[0][n - 1]