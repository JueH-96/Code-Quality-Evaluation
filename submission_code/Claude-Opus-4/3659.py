class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])
        MOD = 10**9 + 7
        
        # dp[i][j][xor_val] = number of paths from (0,0) to (i,j) with XOR value = xor_val
        dp = [[[0 for _ in range(16)] for _ in range(n)] for _ in range(m)]
        
        # Base case: starting position
        dp[0][0][grid[0][0]] = 1
        
        # Fill the DP table
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                    
                # For each possible XOR value
                for xor_val in range(16):
                    # Current cell value
                    curr_val = grid[i][j]
                    
                    # If we came from the cell above (i-1, j)
                    if i > 0:
                        # The previous XOR value would be xor_val ^ curr_val
                        prev_xor = xor_val ^ curr_val
                        dp[i][j][xor_val] = (dp[i][j][xor_val] + dp[i-1][j][prev_xor]) % MOD
                    
                    # If we came from the cell to the left (i, j-1)
                    if j > 0:
                        # The previous XOR value would be xor_val ^ curr_val
                        prev_xor = xor_val ^ curr_val
                        dp[i][j][xor_val] = (dp[i][j][xor_val] + dp[i][j-1][prev_xor]) % MOD
        
        # Return the number of paths to bottom-right with XOR value = k
        return dp[m-1][n-1][k]