from typing import List

class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        mod = 10**9 + 7
        m, n = len(grid), len(grid[0])
        # As grid values are in range [0, 15], max XOR value will be at most 15.
        dp = [[[0] * 16 for _ in range(n)] for _ in range(m)]
        
        dp[0][0][grid[0][0]] = 1  # Starting at (0, 0)
        
        for i in range(m):
            for j in range(n):
                for xor_val in range(16):
                    if dp[i][j][xor_val] != 0:
                        if i + 1 < m:
                            new_xor = xor_val ^ grid[i+1][j]
                            dp[i+1][j][new_xor] = (dp[i+1][j][new_xor] + dp[i][j][xor_val]) % mod
                        if j + 1 < n:
                            new_xor = xor_val ^ grid[i][j+1]
                            dp[i][j+1][new_xor] = (dp[i][j+1][new_xor] + dp[i][j][xor_val]) % mod
        
        return dp[m-1][n-1][k]