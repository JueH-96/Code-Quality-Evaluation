from typing import List

class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        # Cost for making all rows palindromic
        # For each row, for each pair (j, n-1-j),
        # if grid[i][j] != grid[i][n-1-j], we need one flip.
        row_flips = 0
        for i in range(m):
            for j in range(n // 2):
                if grid[i][j] != grid[i][n - j - 1]:
                    row_flips += 1
        
        # Cost for making all columns palindromic
        # For each column, for each pair (i, m-1-i),
        # if grid[i][col] != grid[m-1-i][col], we need one flip.
        col_flips = 0
        for j in range(n):
            for i in range(m // 2):
                if grid[i][j] != grid[m - i - 1][j]:
                    col_flips += 1
        
        return min(row_flips, col_flips)