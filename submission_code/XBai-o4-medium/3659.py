from typing import List

MOD = 10**9 + 7

class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m = len(grid)
        n = len(grid[0])
        
        # Initialize a 3D DP array: dp[i][j][x] represents the number of paths to (i,j) with XOR value x
        dp = [[[0] * 16 for _ in range(n)] for _ in range(m)]
        
        # Base case: starting cell (0,0)
        initial_val = grid[0][0]
        dp[0][0][initial_val] = 1
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                current_val = grid[i][j]
                temp = [0] * 16
                
                # Check top cell (i-1, j)
                if i > 0:
                    for x in range(16):
                        count = dp[i-1][j][x]
                        if count > 0:
                            new_x = x ^ current_val
                            temp[new_x] = (temp[new_x] + count) % MOD
                
                # Check left cell (i, j-1)
                if j > 0:
                    for x in range(16):
                        count = dp[i][j-1][x]
                        if count > 0:
                            new_x = x ^ current_val
                            temp[new_x] = (temp[new_x] + count) % MOD
                
                # Update the dp[i][j] with the calculated temp values
                for x in range(16):
                    dp[i][j][x] = temp[x]
        
        return dp[m-1][n-1][k] % MOD