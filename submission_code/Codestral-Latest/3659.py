class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m, n = len(grid), len(grid[0])

        # Initialize a 3D DP array where dp[i][j][xor_val] represents the number of ways to reach (i, j) with XOR value xor_val
        dp = [[[0] * 16 for _ in range(n)] for _ in range(m)]

        # Start with the top-left cell
        dp[0][0][grid[0][0]] = 1

        # Fill the DP table
        for i in range(m):
            for j in range(n):
                for xor_val in range(16):
                    if dp[i][j][xor_val] > 0:
                        # Move right
                        if j + 1 < n:
                            new_xor = xor_val ^ grid[i][j + 1]
                            dp[i][j + 1][new_xor] = (dp[i][j + 1][new_xor] + dp[i][j][xor_val]) % MOD
                        # Move down
                        if i + 1 < m:
                            new_xor = xor_val ^ grid[i + 1][j]
                            dp[i + 1][j][new_xor] = (dp[i + 1][j][new_xor] + dp[i][j][xor_val]) % MOD

        # The result is the number of ways to reach the bottom-right cell with XOR value k
        return dp[m - 1][n - 1][k]