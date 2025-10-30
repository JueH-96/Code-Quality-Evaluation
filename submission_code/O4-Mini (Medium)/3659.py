from typing import List

class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        mod = 10**9 + 7
        m, n = len(grid), len(grid[0])
        
        # dp[i][j][x] = number of ways to reach cell (i,j) with XOR value x
        dp = [[[0]*16 for _ in range(n)] for __ in range(m)]
        
        # base case: start at (0,0)
        dp[0][0][grid[0][0]] = 1
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                val = grid[i][j]
                cur = dp[i][j]
                
                # from above
                if i > 0:
                    up = dp[i-1][j]
                    for x in range(16):
                        cnt = up[x]
                        if cnt:
                            cur[x ^ val] = (cur[x ^ val] + cnt) % mod
                
                # from left
                if j > 0:
                    left = dp[i][j-1]
                    for x in range(16):
                        cnt = left[x]
                        if cnt:
                            cur[x ^ val] = (cur[x ^ val] + cnt) % mod
        
        return dp[m-1][n-1][k]