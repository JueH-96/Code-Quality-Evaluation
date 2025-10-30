class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])
        MOD = 10**9 + 7
        
        # Initialize 3D DP array: [row][col][xor_value]
        dp = [[[0 for _ in range(16)] for _ in range(n)] for _ in range(m)]
        
        # Base case: one path at the starting cell with its value as XOR
        dp[0][0][grid[0][0]] = 1
        
        # Fill the DP table
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                
                # Process paths coming from above (i-1, j)
                if i > 0:
                    for prev_xor in range(16):
                        if dp[i-1][j][prev_xor] > 0:
                            new_xor = prev_xor ^ grid[i][j]
                            dp[i][j][new_xor] = (dp[i][j][new_xor] + dp[i-1][j][prev_xor]) % MOD
                
                # Process paths coming from left (i, j-1)
                if j > 0:
                    for prev_xor in range(16):
                        if dp[i][j-1][prev_xor] > 0:
                            new_xor = prev_xor ^ grid[i][j]
                            dp[i][j][new_xor] = (dp[i][j][new_xor] + dp[i][j-1][prev_xor]) % MOD
        
        # Return the number of paths to the bottom-right cell with XOR = k
        return dp[m-1][n-1][k]