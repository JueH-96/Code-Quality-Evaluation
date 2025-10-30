from typing import List

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Function to calculate the strength of a subarray
        def calculate_strength(subarray):
            x = len(subarray)
            strength = 0
            for i in range(x):
                strength += ((-1) ** (i + 1)) * sum(subarray[:i + 1]) * (x - i)
            return strength
        
        # Dynamic programming to find the maximum strength
        dp = [[-float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                for l in range(i):
                    subarray = nums[l:i]
                    dp[i][j] = max(dp[i][j], dp[l][j - 1] + calculate_strength(subarray))
        
        return dp[n][k]