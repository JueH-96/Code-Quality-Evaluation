class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m, n = len(grid), len(grid[0])
        
        # dp[i][j][x] will be the number of ways to reach cell (i, j) with XOR value x
        dp = [[[0 for _ in range(16)] for _ in range(n)] for _ in range(m)]
        
        # Initialize the starting position
        dp[0][0][grid[0][0]] = 1
        
        # Fill the dp table
        for i in range(m):
            for j in range(n):
                for x in range(16):  # XOR values can only be between 0 and 15
                    if i == 0 and j == 0:
                        continue
                    # From top
                    if i > 0:
                        dp[i][j][x] += dp[i-1][j][x ^ grid[i][j]]
                        dp[i][j][x] %= MOD
                    # From left
                    if j > 0:
                        dp[i][j][x] += dp[i][j-1][x ^ grid[i][j]]
                        dp[i][j][x] %= MOD
        
        # The answer is the number of paths that reach the bottom-right corner with XOR equal to k
        return dp[m-1][n-1][k]