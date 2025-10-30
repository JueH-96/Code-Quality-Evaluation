class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m, n = len(grid), len(grid[0])
        
        # dp[i][j][x] will store the number of ways to reach cell (i, j)
        # with XOR value x (including grid[i][j]).
        dp = [[[0]*16 for _ in range(n)] for __ in range(m)]
        
        # Initialize for the top-left cell
        dp[0][0][grid[0][0]] = 1
        
        # Fill the dp table
        for i in range(m):
            for j in range(n):
                for x in range(16):
                    ways = dp[i][j][x]
                    if ways == 0:
                        continue
                    # Move down
                    if i + 1 < m:
                        new_x = x ^ grid[i+1][j]
                        dp[i+1][j][new_x] = (dp[i+1][j][new_x] + ways) % MOD
                    # Move right
                    if j + 1 < n:
                        new_x = x ^ grid[i][j+1]
                        dp[i][j+1][new_x] = (dp[i][j+1][new_x] + ways) % MOD
        
        return dp[m-1][n-1][k] % MOD