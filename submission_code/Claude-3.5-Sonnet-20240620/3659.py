class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m, n = len(grid), len(grid[0])
        
        # Initialize 3D DP array
        dp = [[[0] * 16 for _ in range(n)] for _ in range(m)]
        
        # Base case: top-left cell
        dp[0][0][grid[0][0]] = 1
        
        # Fill the DP array
        for i in range(m):
            for j in range(n):
                for xor in range(16):
                    if i > 0:
                        dp[i][j][xor ^ grid[i][j]] = (dp[i][j][xor ^ grid[i][j]] + dp[i-1][j][xor]) % MOD
                    if j > 0:
                        dp[i][j][xor ^ grid[i][j]] = (dp[i][j][xor ^ grid[i][j]] + dp[i][j-1][xor]) % MOD
        
        # Return the count of paths with XOR value k
        return dp[m-1][n-1][k]