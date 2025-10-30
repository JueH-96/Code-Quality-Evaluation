class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Precompute costs for all subarrays
        costs = [[0] * n for _ in range(n)]
        for l in range(n):
            costs[l][l] = nums[l]  # Single element subarray
            for r in range(l+1, n):
                # Extend the subarray by one element, alternating sign
                costs[l][r] = costs[l][r-1] + nums[r] * ((-1) ** (r - l))
        
        # Initialize DP array - dp[i] represents max cost using elements 0 to i
        dp = [0] * n
        dp[0] = nums[0]
        
        # Fill the DP array
        for i in range(1, n):
            # Option 1: No split (use the whole subarray from 0 to i)
            dp[i] = costs[0][i]
            
            # Option 2: Try all possible split points
            for j in range(i):
                dp[i] = max(dp[i], dp[j] + costs[j+1][i])
        
        return dp[n-1]