from typing import List

class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])

        # Calculate flips required to make all rows palindromic
        flips_rows = 0
        for row in grid:
            for j in range(n // 2):
                if row[j] != row[n - 1 - j]:
                    flips_rows += 1

        # Calculate flips required to make all columns palindromic
        flips_cols = 0
        for col in range(n):
            for i in range(m // 2):
                if grid[i][col] != grid[m - 1 - i][col]:
                    flips_cols += 1

        return min(flips_rows, flips_cols)