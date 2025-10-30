from typing import List

class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m = len(grid)
        if m == 0:
            return 0
        n = len(grid[0])
        
        # Cost to make all rows palindromic
        cost_rows = 0
        half_n = n // 2
        for i in range(m):
            row = grid[i]
            # Compare pairs (j, n-1-j)
            for j in range(half_n):
                if row[j] != row[n - 1 - j]:
                    cost_rows += 1
        
        # Cost to make all columns palindromic
        cost_cols = 0
        half_m = m // 2
        for j in range(n):
            # Compare pairs (i, m-1-i)
            for i in range(half_m):
                if grid[i][j] != grid[m - 1 - i][j]:
                    cost_cols += 1
        
        return min(cost_rows, cost_cols)