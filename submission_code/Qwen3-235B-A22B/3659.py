from typing import List

class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m = len(grid)
        n = len(grid[0])
        dp = [[[0] * 16 for _ in range(n)] for _ in range(m)]
        
        # Initialize the starting cell
        dp[0][0][grid[0][0]] = 1
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                current_val = grid[i][j]
                temp = [0] * 16
                # Check top cell
                if i > 0:
                    up = dp[i-1][j]
                    for prev_xor in range(16):
                        count = up[prev_xor]
                        if count:
                            new_xor = prev_xor ^ current_val
                            temp[new_xor] = (temp[new_xor] + count) % MOD
                # Check left cell
                if j > 0:
                    left = dp[i][j-1]
                    for prev_xor in range(16):
                        count = left[prev_xor]
                        if count:
                            new_xor = prev_xor ^ current_val
                            temp[new_xor] = (temp[new_xor] + count) % MOD
                dp[i][j] = temp
        
        return dp[m-1][n-1][k] % MOD