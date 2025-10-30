from typing import List

class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        row_flips = 0
        for row in grid:
            n = len(row)
            for i in range(n // 2):
                j = n - 1 - i
                if row[i] != row[j]:
                    row_flips += 1
        
        m = len(grid)
        n = len(grid[0])
        column_flips = 0
        for col in range(n):
            for i in range(m // 2):
                j = m - 1 - i
                if grid[i][col] != grid[j][col]:
                    column_flips += 1
        
        return min(row_flips, column_flips)