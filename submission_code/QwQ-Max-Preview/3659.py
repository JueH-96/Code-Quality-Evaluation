from typing import List

MOD = 10**9 + 7

class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m = len(grid)
        n = len(grid[0])
        dp = [[[0] * 16 for _ in range(n)] for __ in range(m)]
        
        # Initialize the starting cell (0, 0)
        initial_val = grid[0][0]
        dp[0][0][initial_val] = 1
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue  # Skip the initial cell
                current_val = grid[i][j]
                current_dp = [0] * 16
                # Check the top cell (i-1, j)
                if i > 0:
                    for x in range(16):
                        if dp[i-1][j][x]:
                            new_xor = x ^ current_val
                            current_dp[new_xor] = (current_dp[new_xor] + dp[i-1][j][x]) % MOD
                # Check the left cell (i, j-1)
                if j > 0:
                    for x in range(16):
                        if dp[i][j-1][x]:
                            new_xor = x ^ current_val
                            current_dp[new_xor] = (current_dp[new_xor] + dp[i][j-1][x]) % MOD
                # Update the current cell's DP
                dp[i][j] = current_dp
        
        return dp[m-1][n-1][k] % MOD