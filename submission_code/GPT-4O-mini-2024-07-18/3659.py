class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m, n = len(grid), len(grid[0])
        
        # dp[i][j][xor_value] will store the number of ways to reach (i, j) with a specific xor_value
        dp = [[[0] * 16 for _ in range(n)] for _ in range(m)]
        
        # Initialize the starting point
        dp[0][0][grid[0][0]] = 1
        
        for i in range(m):
            for j in range(n):
                for xor_value in range(16):
                    if dp[i][j][xor_value] > 0:
                        # Move down
                        if i + 1 < m:
                            new_xor = xor_value ^ grid[i + 1][j]
                            dp[i + 1][j][new_xor] = (dp[i + 1][j][new_xor] + dp[i][j][xor_value]) % MOD
                        
                        # Move right
                        if j + 1 < n:
                            new_xor = xor_value ^ grid[i][j + 1]
                            dp[i][j + 1][new_xor] = (dp[i][j + 1][new_xor] + dp[i][j][xor_value]) % MOD
        
        # The answer is the number of ways to reach the bottom-right corner with the required xor value
        return dp[m - 1][n - 1][k]