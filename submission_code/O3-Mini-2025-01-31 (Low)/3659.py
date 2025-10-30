from typing import List

MOD = 10**9 + 7

class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])
        # dp[i][j][x] will represent count of ways to reach cell (i,j) with xor value x
        dp = [[[0] * 16 for _ in range(n)] for _ in range(m)]
        dp[0][0][grid[0][0]] = 1
        
        for i in range(m):
            for j in range(n):
                # For each possible xor value at this cell
                for xor_val in range(16):
                    count = dp[i][j][xor_val]
                    if count:
                        # Move Down
                        if i + 1 < m:
                            new_xor = xor_val ^ grid[i+1][j]
                            dp[i+1][j][new_xor] = (dp[i+1][j][new_xor] + count) % MOD
                        # Move Right
                        if j + 1 < n:
                            new_xor = xor_val ^ grid[i][j+1]
                            dp[i][j+1][new_xor] = (dp[i][j+1][new_xor] + count) % MOD
                            
        return dp[m-1][n-1][k] % MOD