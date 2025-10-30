class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m, n = len(grid), len(grid[0])
        
        # dp[i][j][xor] = number of paths from (0,0) to (i,j) with XOR value = xor
        dp = [[[0 for _ in range(16)] for _ in range(n)] for _ in range(m)]
        
        # Base case
        dp[0][0][grid[0][0]] = 1
        
        # Fill the DP table
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                
                # From top cell (i-1, j)
                if i > 0:
                    for prev_xor in range(16):
                        new_xor = prev_xor ^ grid[i][j]
                        dp[i][j][new_xor] = (dp[i][j][new_xor] + dp[i-1][j][prev_xor]) % MOD
                
                # From left cell (i, j-1)
                if j > 0:
                    for prev_xor in range(16):
                        new_xor = prev_xor ^ grid[i][j]
                        dp[i][j][new_xor] = (dp[i][j][new_xor] + dp[i][j-1][prev_xor]) % MOD
        
        return dp[m-1][n-1][k]