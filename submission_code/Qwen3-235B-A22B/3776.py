from typing import List

class Solution:
    def minCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        dp = [0] * (n + 3)  # Initialize a DP array with enough size to handle i+3
        
        # Base cases
        if n >= 1:
            dp[n-1] = nums[n-1]
        if n >= 2:
            dp[n-2] = max(nums[n-2], nums[n-1])
        
        # Fill dp for i from n-3 down to 0
        for i in range(n-3, -1, -1):
            # Three options:
            # 1. Remove first and second elements
            cost1 = max(nums[i], nums[i+1]) + dp[i+2]
            # 2. Remove first and third elements
            cost2 = max(nums[i], nums[i+2]) + dp[i+1]
            # 3. Remove second and third elements
            cost3 = max(nums[i+1], nums[i+2]) + (dp[i+3] if i+3 < n else 0)
            dp[i] = min(cost1, cost2, cost3)
        
        return dp[0]