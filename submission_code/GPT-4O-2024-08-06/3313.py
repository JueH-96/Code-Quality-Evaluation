from typing import List

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Helper function to calculate the strength of a subarray
        def calculate_strength(subarray: List[int]) -> int:
            x = len(subarray)
            total_strength = 0
            for i in range(1, x + 1):
                sub_sum = sum(subarray[:i])
                total_strength += ((-1) ** (i + 1)) * sub_sum * (x - i + 1)
            return total_strength
        
        # Dynamic programming approach to find the maximum strength
        # dp[i][j] will store the maximum strength using the first i elements with j subarrays
        dp = [[float('-inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        # Iterate over the array
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                # Consider all possible subarray endings at i
                for l in range(j, i + 1):
                    subarray = nums[l-1:i]
                    strength = calculate_strength(subarray)
                    dp[i][j] = max(dp[i][j], dp[l-1][j-1] + strength)
        
        return dp[n][k]