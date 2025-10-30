class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        # dp[i][j] = max score selecting i elements from first j elements of b
        # Initialize with negative infinity
        dp = [[-float('inf')] * (n + 1) for _ in range(5)]
        
        # Base case: selecting 0 elements gives score 0
        for j in range(n + 1):
            dp[0][j] = 0
        
        # Fill the dp table
        for i in range(1, 5):  # We need to select 1 to 4 elements
            for j in range(i, n + 1):  # Need at least i elements to select i
                # Option 1: Don't include b[j-1]
                dp[i][j] = dp[i][j-1]
                
                # Option 2: Include b[j-1] as the i-th selection
                if j >= i:  # Can only include if we have enough elements
                    dp[i][j] = max(dp[i][j], dp[i-1][j-1] + a[i-1] * b[j-1])
        
        return dp[4][n]