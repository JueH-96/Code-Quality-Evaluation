class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m = len(grid)
        n = len(grid[0]) if m else 0
        # Initialize a 3D DP array where dp[i][j][x] is the number of ways to reach (i,j) with XOR x
        dp = [[[0] * 16 for _ in range(n)] for __ in range(m)]
        # Base case: starting cell
        dp[0][0][grid[0][0]] = 1
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue  # already initialized
                current_val = grid[i][j]
                # Initialize current cell's DP with zeros (by default it's already zero)
                # Contributions from the top cell (i-1, j)
                if i > 0:
                    for x in range(16):
                        new_x = x ^ current_val
                        dp[i][j][new_x] = (dp[i][j][new_x] + dp[i-1][j][x]) % MOD
                # Contributions from the left cell (i, j-1)
                if j > 0:
                    for x in range(16):
                        new_x = x ^ current_val
                        dp[i][j][new_x] = (dp[i][j][new_x] + dp[i][j-1][x]) % MOD
        
        return dp[m-1][n-1][k] % MOD