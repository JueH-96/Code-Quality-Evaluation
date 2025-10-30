class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Create a 2D dynamic programming array to store the maximum value for each subsequence of length 2*k
        dp = [[0] * (2*k) for _ in range(n)]
        
        # Initialize the first row of the DP array
        for i in range(2*k):
            dp[0][i] = nums[0]
        
        # Fill the DP array
        for i in range(1, n):
            for j in range(2*k):
                # If the current index is less than 2*k, use the previous maximum value
                if j < 2*k:
                    dp[i][j] = max(dp[i-1][j], nums[i])
                # Otherwise, calculate the maximum value for the current subsequence
                else:
                    dp[i][j] = max(dp[i-1][j], (dp[i-1][j-1] | nums[i]) ^ (dp[i-1][j-1] | nums[i]))
        
        # Return the maximum value in the last row of the DP array
        return dp[n-1][2*k-1]