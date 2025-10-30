class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [[[float('inf')] * 3 for _ in range(n)] for _ in range(n)]
        
        # Base case: one subarray
        for i in range(n):
            dp[i][i][0] = nums[i]
        
        # Fill the dp table
        for length in range(2, n+1):
            for i in range(n-length+1):
                j = i + length - 1
                for k in range(i, j):
                    dp[i][j][0] = min(dp[i][j][0], dp[i][k][0] + dp[k+1][j][0])
                    dp[i][j][1] = min(dp[i][j][1], dp[i][k][1] + dp[k+1][j][1])
                    dp[i][j][2] = min(dp[i][j][2], dp[i][k][2] + dp[k+1][j][2], dp[i][k][0] + dp[k+1][j][1] + dp[k+1][j][2])
        
        return min(dp[0][n-1])