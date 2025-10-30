class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Initialize dp array
        dp = [[float('inf')] * 4 for _ in range(n + 1)]
        dp[0][0] = 0
        
        # Fill dp array
        for i in range(1, n + 1):
            for j in range(1, min(4, i + 1)):
                for k in range(j - 1, i):
                    dp[i][j] = min(dp[i][j], dp[k][j-1] + nums[k])
        
        # Return the minimum cost for 3 subarrays
        return dp[n][3]