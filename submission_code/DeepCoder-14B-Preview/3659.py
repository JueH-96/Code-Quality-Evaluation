class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m = len(grid)
        if m == 0:
            return 0
        n = len(grid[0])
        if n == 0:
            return 0
        
        # Initialize DP table
        dp = [[[0] * 16 for _ in range(n)] for __ in range(m)]
        dp[0][0][grid[0][0]] = 1  # Starting cell's XOR is grid[0][0]
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue  # Skip the starting cell as it's already initialized
                # Contributions from the top cell (i-1, j)
                if i > 0:
                    for x_prev in range(16):
                        ways = dp[i-1][j][x_prev]
                        if ways == 0:
                            continue
                        new_x = x_prev ^ grid[i][j]
                        dp[i][j][new_x] = (dp[i][j][new_x] + ways) % MOD
                # Contributions from the left cell (i, j-1)
                if j > 0:
                    for x_prev in range(16):
                        ways = dp[i][j-1][x_prev]
                        if ways == 0:
                            continue
                        new_x = x_prev ^ grid[i][j]
                        dp[i][j][new_x] = (dp[i][j][new_x] + ways) % MOD
        
        return dp[m-1][n-1][k] % MOD