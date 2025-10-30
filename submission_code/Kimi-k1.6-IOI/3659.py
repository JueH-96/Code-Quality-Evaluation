class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m = len(grid)
        n = len(grid[0]) if m else 0
        
        # Initialize a DP table with dimensions m x n x 16
        dp = [[[0] * 16 for _ in range(n)] for __ in range(m)]
        
        # Base case: starting cell (0, 0)
        dp[0][0][grid[0][0]] = 1
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue  # already initialized
                current_val = grid[i][j]
                for x in range(16):
                    total = 0
                    # Check top cell (i-1, j)
                    if i > 0:
                        prev_x = x ^ current_val
                        total += dp[i-1][j][prev_x]
                    # Check left cell (i, j-1)
                    if j > 0:
                        prev_x = x ^ current_val
                        total += dp[i][j-1][prev_x]
                    dp[i][j][x] = total % MOD
        
        return dp[m-1][n-1][k] % MOD