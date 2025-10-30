class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m, n = len(grid), len(grid[0])
        
        # dp[i][j][xor_val] = number of ways to reach (i,j) with XOR value xor_val
        dp = [[[0 for _ in range(16)] for _ in range(n)] for _ in range(m)]
        
        # Base case: starting position
        dp[0][0][grid[0][0]] = 1
        
        # Fill the DP table
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                
                for xor_val in range(16):
                    # Coming from top
                    if i > 0:
                        prev_xor = xor_val ^ grid[i][j]
                        dp[i][j][xor_val] = (dp[i][j][xor_val] + dp[i-1][j][prev_xor]) % MOD
                    
                    # Coming from left
                    if j > 0:
                        prev_xor = xor_val ^ grid[i][j]
                        dp[i][j][xor_val] = (dp[i][j][xor_val] + dp[i][j-1][prev_xor]) % MOD
        
        return dp[m-1][n-1][k]