from typing import List

class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        def row_flips():
            flips = 0
            for row in grid:
                n = len(row)
                for i in range(n // 2):
                    if row[i] != row[n - 1 - i]:
                        flips += 1
            return flips
        
        def col_flips():
            flips = 0
            m = len(grid)
            n = len(grid[0]) if m > 0 else 0
            for j in range(n):
                for i in range(m // 2):
                    if grid[i][j] != grid[m - 1 - i][j]:
                        flips += 1
            return flips
        
        return min(row_flips(), col_flips())