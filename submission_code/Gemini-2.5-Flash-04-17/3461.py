from typing import List

class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        """
        Finds the area of the smallest axis-aligned rectangle that contains all 1s.

        Args:
            grid: A 2D binary array.

        Returns:
            The minimum possible area of the bounding rectangle.
        """
        # Get the dimensions of the grid
        m = len(grid)
        n = len(grid[0])

        # Initialize variables to track the boundaries of the 1s.
        # min_row and min_col are initialized to values larger than any possible index.
        # max_row and max_col are initialized to values smaller than any possible index.
        # These serve as sentinels that will be updated by the first 1 encountered.
        min_row = m
        max_row = -1
        min_col = n
        max_col = -1

        # Iterate through each cell of the grid
        for r in range(m):
            for c in range(n):
                # If the cell contains a 1
                if grid[r][c] == 1:
                    # Update the boundary indices to include this 1
                    min_row = min(min_row, r)
                    max_row = max(max_row, r)
                    min_col = min(min_col, c)
                    max_col = max(max_col, c)

        # Calculate the height and width of the smallest bounding rectangle.
        # The height is the difference between the maximum and minimum row indices + 1.
        # The width is the difference between the maximum and minimum column indices + 1.
        # The problem guarantees at least one 1, so min_row, max_row, min_col,
        # and max_col will have valid indices after the loop (min <= max).
        height = max_row - min_row + 1
        width = max_col - min_col + 1

        # The minimum area is the product of the height and width.
        return height * width