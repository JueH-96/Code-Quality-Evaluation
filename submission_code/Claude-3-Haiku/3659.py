class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])
        MOD = 10 ** 9 + 7
        
        # Create a 3D array to store the number of paths
        # dp[i][j][x] represents the number of paths from (0, 0) to (i, j)
        # with the XOR value of the path being x
        dp = [[[0] * 16 for _ in range(n)] for _ in range(m)]
        
        # Initialize the first cell
        dp[0][0][grid[0][0]] = 1
        
        # Iterate through the grid
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                
                # If we can move to the right
                if j > 0:
                    for x in range(16):
                        dp[i][j][x ^ grid[i][j]] = (dp[i][j][x ^ grid[i][j]] + dp[i][j-1][x]) % MOD
                
                # If we can move down
                if i > 0:
                    for x in range(16):
                        dp[i][j][x ^ grid[i][j]] = (dp[i][j][x ^ grid[i][j]] + dp[i-1][j][x]) % MOD
        
        # Return the number of paths with XOR value equal to k
        return dp[m-1][n-1][k]