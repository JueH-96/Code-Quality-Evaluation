import math
from typing import List

class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        """
        Calculates the minimum number of flips to make either all rows or all columns palindromic in a binary matrix.

        A row or column is palindromic if its values read the same forwards and backward.
        We can flip any cell's value (0 to 1 or 1 to 0).
        The function computes the cost (number of flips) to make all rows palindromic
        and the cost to make all columns palindromic, then returns the minimum of these two costs.

        Args:
            grid: An m x n binary matrix (list of lists of integers 0 or 1).

        Returns:
            The minimum number of flips required to satisfy the condition.
        """
        
        m = len(grid)
        # Constraints state m >= 1, so this check is technically redundant but safe.
        # If the problem allowed empty grids, this would be necessary.
        # if m == 0:
        #     return 0 
            
        n = len(grid[0])
        # Constraints state n >= 1 (since m*n >= 1), so this check is also redundant.
        # if n == 0:
        #     return 0

        # --- Scenario 1: Calculate flips needed to make all rows palindromic ---
        flips_rows = 0
        # Iterate through each row of the grid.
        for r in range(m):
            # For each row, we check pairs of elements from the start and end, moving inwards.
            # We only need to iterate up to the middle index of the row.
            # Integer division `n // 2` handles both even and odd lengths correctly.
            # For odd length rows, the middle element doesn't need comparison as it matches itself.
            for c in range(n // 2):
                # Compare the element at index `c` with the element at the symmetric index `n - 1 - c`.
                if grid[r][c] != grid[r][n - 1 - c]:
                    # If they are different, one flip is required for this pair to make them match.
                    # We add 1 to the total flips count for making rows palindromic.
                    # This counts the minimum flips needed because flipping either cell makes them equal.
                    flips_rows += 1

        # --- Scenario 2: Calculate flips needed to make all columns palindromic ---
        flips_cols = 0
        # Iterate through each column of the grid.
        for c in range(n):
            # For each column, we check pairs of elements from the top and bottom, moving inwards.
            # We only need to iterate up to the middle index of the column.
            # Integer division `m // 2` handles both even and odd heights correctly.
            # For odd height columns, the middle element doesn't need comparison.
            for r in range(m // 2):
                # Compare the element at index `r` with the element at the symmetric index `m - 1 - r`.
                if grid[r][c] != grid[m - 1 - r][c]:
                    # If they are different, one flip is required for this pair to make them match.
                    # We add 1 to the total flips count for making columns palindromic.
                    flips_cols += 1

        # The final result is the minimum number of flips required between the two scenarios:
        # either making all rows palindromic or making all columns palindromic.
        return min(flips_rows, flips_cols)