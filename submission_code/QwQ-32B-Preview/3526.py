from typing import List

class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        
        # Calculate flips needed to make all rows palindromic
        flips_rows = 0
        for i in range(m):
            for j in range(n // 2):
                if grid[i][j] != grid[i][n - j - 1]:
                    flips_rows += 1
        
        # Calculate flips needed to make all columns palindromic
        flips_columns = 0
        for k in range(n):
            for j in range(m // 2):
                if grid[j][k] != grid[m - j - 1][k]:
                    flips_columns += 1
        
        # Return the minimum of the two flip counts
        return min(flips_rows, flips_columns)