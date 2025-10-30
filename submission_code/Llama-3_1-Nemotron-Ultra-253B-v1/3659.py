from typing import List

MOD = 10**9 + 7

class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m = len(grid)
        n = len(grid[0]) if m else 0
        if m == 0 or n == 0:
            return 0
        
        # Initialize DP table
        dp = [[[0] * 16 for _ in range(n)] for __ in range(m)]
        dp[0][0][grid[0][0]] = 1
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                current = grid[i][j]
                for x in range(16):
                    from_top = 0
                    if i > 0:
                        prev_x = x ^ current
                        from_top = dp[i-1][j][prev_x]
                    from_left = 0
                    if j > 0:
                        prev_x = x ^ current
                        from_left = dp[i][j-1][prev_x]
                    dp[i][j][x] = (from_top + from_left) % MOD
        
        return dp[m-1][n-1][k] % MOD