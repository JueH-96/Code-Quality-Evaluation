class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m, n = len(grid), len(grid[0])
        
        # dp[i][j][x] = number of ways to reach cell (i, j) with XOR sum = x
        dp = [[[0]*16 for _ in range(n)] for _ in range(m)]
        
        # Initialize starting cell
        dp[0][0][grid[0][0]] = 1
        
        for i in range(m):
            for j in range(n):
                for x_val in range(16):
                    ways = dp[i][j][x_val]
                    if ways == 0:
                        continue
                    
                    # Move down
                    if i + 1 < m:
                        new_xor = x_val ^ grid[i+1][j]
                        dp[i+1][j][new_xor] = (dp[i+1][j][new_xor] + ways) % MOD
                    
                    # Move right
                    if j + 1 < n:
                        new_xor = x_val ^ grid[i][j+1]
                        dp[i][j+1][new_xor] = (dp[i][j+1][new_xor] + ways) % MOD
        
        return dp[m-1][n-1][k] % MOD