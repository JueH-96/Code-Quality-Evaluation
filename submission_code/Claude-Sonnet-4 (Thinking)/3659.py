class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])
        MOD = 10**9 + 7
        
        # dp[i][j][x] = number of paths from (0, 0) to (i, j) with XOR value x
        dp = [[[0] * 16 for _ in range(n)] for _ in range(m)]
        
        # Base case
        dp[0][0][grid[0][0]] = 1
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                
                for x in range(16):
                    # Coming from top
                    if i > 0:
                        prev_xor = x ^ grid[i][j]
                        dp[i][j][x] = (dp[i][j][x] + dp[i-1][j][prev_xor]) % MOD
                    
                    # Coming from left
                    if j > 0:
                        prev_xor = x ^ grid[i][j]
                        dp[i][j][x] = (dp[i][j][x] + dp[i][j-1][prev_xor]) % MOD
        
        return dp[m-1][n-1][k]