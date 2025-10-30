from typing import List

class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m, n = len(grid), len(grid[0])
        # dp_prev[j][x] = number of ways to reach cell (i-1, j) with xor = x
        dp_prev = [[0]*16 for _ in range(n)]
        
        for i in range(m):
            # dp_curr[j][x] = number of ways to reach cell (i, j) with xor = x
            dp_curr = [[0]*16 for _ in range(n)]
            for j in range(n):
                v = grid[i][j]
                if i == 0 and j == 0:
                    # start at (0,0)
                    dp_curr[j][v] = 1
                else:
                    # take from above
                    if i > 0:
                        above = dp_prev[j]
                        for x in range(16):
                            cnt = above[x]
                            if cnt:
                                dp_curr[j][x ^ v] = (dp_curr[j][x ^ v] + cnt) % MOD
                    # take from left
                    if j > 0:
                        left = dp_curr[j-1]
                        for x in range(16):
                            cnt = left[x]
                            if cnt:
                                dp_curr[j][x ^ v] = (dp_curr[j][x ^ v] + cnt) % MOD
            dp_prev = dp_curr
        
        return dp_prev[n-1][k]