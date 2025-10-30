from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        
        # Precompute the prefix sums
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        # Function to get the sum of a subarray from l to r (inclusive)
        def get_sum(l: int, r: int) -> int:
            return prefix_sum[r + 1] - prefix_sum[l]
        
        # dp[i][j] will store the maximum sum we can get with j subarrays using the first i elements
        dp = [[float('-inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0  # Base case: 0 subarrays from 0 elements has sum 0
        
        for j in range(1, k + 1):  # for each number of subarrays
            for i in range(m * j, n + 1):  # for each position in nums
                # We need to consider the last subarray starting from any position `p`
                for p in range(i - m, -1, -1):  # p is the start of the last subarray
                    dp[i][j] = max(dp[i][j], dp[p][j - 1] + get_sum(p, i - 1))
        
        # The answer will be the maximum sum we can get with k subarrays using all elements
        return max(dp[i][k] for i in range(n + 1))