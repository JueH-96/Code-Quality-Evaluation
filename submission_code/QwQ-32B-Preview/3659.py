class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m = len(grid)
        n = len(grid[0])
        
        # Initialize DP array with zeros
        dp = [[[0 for _ in range(16)] for _ in range(n)] for _ in range(m)]
        
        # Starting point
        dp[0][0][grid[0][0]] = 1
        
        # Fill DP array
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue  # Already initialized
                for xor_val in range(16):
                    prev_xor = xor_val ^ grid[i][j]
                    if i > 0:
                        dp[i][j][xor_val] += dp[i-1][j][prev_xor]
                        dp[i][j][xor_val] %= MOD
                    if j > 0:
                        dp[i][j][xor_val] += dp[i][j-1][prev_xor]
                        dp[i][j][xor_val] %= MOD
        
        # Final result
        return dp[m-1][n-1][k]