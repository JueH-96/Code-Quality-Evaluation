from typing import List

class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])
        MOD = 10**9 + 7

        # dp[i][j][x] will store the number of ways to reach cell (i, j) with XOR value x
        dp = [[[0] * 16 for _ in range(n)] for _ in range(m)]

        # Initialize the starting point
        dp[0][0][grid[0][0]] = 1

        # Fill the dp table
        for i in range(m):
            for j in range(n):
                for x in range(16):
                    if dp[i][j][x] > 0:
                        if i + 1 < m:
                            dp[i + 1][j][x ^ grid[i + 1][j]] = (dp[i + 1][j][x ^ grid[i + 1][j]] + dp[i][j][x]) % MOD
                        if j + 1 < n:
                            dp[i][j + 1][x ^ grid[i][j + 1]] = (dp[i][j + 1][x ^ grid[i][j + 1]] + dp[i][j][x]) % MOD

        # The result is the number of ways to reach the bottom-right corner with XOR value k
        return dp[m - 1][n - 1][k]