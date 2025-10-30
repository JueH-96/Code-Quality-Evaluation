class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        from typing import List
        MOD = 10**9 + 7
        
        m = len(grid)
        n = len(grid[0])
        
        # dp[i][j][x] will hold the number of ways to reach cell (i, j)
        # such that the XOR of all values on that path is x.
        dp = [[[0]*16 for _ in range(n)] for _ in range(m)]
        
        # Base case: starting at (0,0)
        dp[0][0][grid[0][0]] = 1
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                val = grid[i][j]
                
                # We want dp[i][j][x] = sum of
                #   dp[i-1][j][x ^ val] (if i > 0) +
                #   dp[i][j-1][x ^ val] (if j > 0)
                # Because from those states x' ^ val = x.
                for x in range(16):
                    prev_x = x ^ val
                    if i > 0:
                        dp[i][j][x] = (dp[i][j][x] + dp[i-1][j][prev_x]) % MOD
                    if j > 0:
                        dp[i][j][x] = (dp[i][j][x] + dp[i][j-1][prev_x]) % MOD
        
        return dp[m-1][n-1][k] % MOD