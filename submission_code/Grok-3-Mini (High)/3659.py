from typing import List

class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m = len(grid)
        n = len(grid[0])
        MOD = 1000000007
        dp = [[[0 for _ in range(16)] for _ in range(n)] for _ in range(m)]
        
        # Set starting point
        dp[0][0][grid[0][0]] = 1
        
        # Fill first row
        for j in range(1, n):
            val = grid[0][j]
            for xor_val in range(16):
                dp[0][j][xor_val] = dp[0][j-1][xor_val ^ val]
                dp[0][j][xor_val] %= MOD
        
        # Fill first column
        for i in range(1, m):
            val = grid[i][0]
            for xor_val in range(16):
                dp[i][0][xor_val] = dp[i-1][0][xor_val ^ val]
                dp[i][0][xor_val] %= MOD
        
        # Fill the rest of the grid
        for i in range(1, m):
            for j in range(1, n):
                val = grid[i][j]
                for xor_val in range(16):
                    prev_xor = xor_val ^ val
                    # Add from left
                    dp[i][j][xor_val] += dp[i][j-1][prev_xor]
                    # Add from above
                    dp[i][j][xor_val] += dp[i-1][j][prev_xor]
                    dp[i][j][xor_val] %= MOD
        
        # Return the number of ways to reach (m-1, n-1) with XOR equal to k
        return dp[m-1][n-1][k]