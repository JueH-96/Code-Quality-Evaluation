from typing import List

class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        # Calculate the number of flips needed to fix all rows
        flips_rows = 0
        for row in grid:
            # Only need to consider the first half of the row
            for j in range(n // 2):
                if row[j] != row[n - 1 - j]:
                    flips_rows += 1
        
        # Calculate the number of flips needed to fix all columns
        flips_cols = 0
        for j in range(n):
            # Only need to consider the first half of the column
            for i in range(m // 2):
                if grid[i][j] != grid[m - 1 - i][j]:
                    flips_cols += 1
        
        # Return the minimum flips needed between the two strategies.
        return min(flips_rows, flips_cols)