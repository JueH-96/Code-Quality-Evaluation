from typing import List

class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])

        # Calculate flips needed to make all rows palindromic
        row_flips = 0
        # Iterate through each row
        for r in range(m):
            # Compare elements from the start and end of the row, moving inwards
            # We only need to iterate up to n // 2 because we compare pairs
            for c in range(n // 2):
                # If the symmetric elements are different, one flip is required for this pair
                if grid[r][c] != grid[r][n - 1 - c]:
                    row_flips += 1

        # Calculate flips needed to make all columns palindromic
        col_flips = 0
        # Iterate through each column
        for c in range(n):
            # Compare elements from the top and bottom of the column, moving inwards
            # We only need to iterate up to m // 2 because we compare pairs
            for r in range(m // 2):
                # If the symmetric elements are different, one flip is required for this pair
                if grid[r][c] != grid[m - 1 - r][c]:
                    col_flips += 1

        # The minimum flips required is the minimum of the flips needed for rows vs columns
        return min(row_flips, col_flips)