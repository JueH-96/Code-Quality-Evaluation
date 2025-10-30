from typing import List

class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m = len(grid)
        if m == 0:
            return 0
        n = len(grid[0])
        
        row_flips = 0
        # Calculate flips needed to make all rows palindromic
        for row in grid:
            for j in range(n // 2):
                if row[j] != row[n - 1 - j]:
                    row_flips += 1
        
        col_flips = 0
        # Calculate flips needed to make all columns palindromic
        for j in range(n):
            for i in range(m // 2):
                if grid[i][j] != grid[m - 1 - i][j]:
                    col_flips += 1
        
        return min(row_flips, col_flips)