class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Create a 2D dynamic programming array to store the maximum value
        # that can be obtained by applying the operation at most i times
        # on the first j elements of the array.
        dp = [[0] * (k + 1) for _ in range(n)]
        
        # Initialize the first row of the DP array
        for i in range(k + 1):
            dp[0][i] = nums[0] << i
        
        # Fill the DP array
        for i in range(1, n):
            for j in range(k + 1):
                # If we don't apply the operation on the current element
                dp[i][j] = max(dp[i][j], dp[i - 1][j] | nums[i])
                # If we apply the operation on the current element
                if j > 0:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] | (nums[i] << 1))
        
        # Return the maximum value in the last row of the DP array
        return dp[n - 1][k]