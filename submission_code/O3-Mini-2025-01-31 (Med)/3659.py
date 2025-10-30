from typing import List

class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        mod = 10**9 + 7
        m, n = len(grid), len(grid[0])
        # Initialize dp as a 2D list where each cell has a list of 16 elements (for XOR values 0 to 15)
        dp = [[[0] * 16 for _ in range(n)] for _ in range(m)]
        dp[0][0][grid[0][0]] = 1
        
        for i in range(m):
            for j in range(n):
                for xor_val in range(16):
                    ways = dp[i][j][xor_val]
                    if ways:
                        # Move Down
                        if i + 1 < m:
                            new_xor = xor_val ^ grid[i + 1][j]
                            dp[i + 1][j][new_xor] = (dp[i + 1][j][new_xor] + ways) % mod
                        # Move Right
                        if j + 1 < n:
                            new_xor = xor_val ^ grid[i][j + 1]
                            dp[i][j + 1][new_xor] = (dp[i][j + 1][new_xor] + ways) % mod
        
        return dp[m - 1][n - 1][k]