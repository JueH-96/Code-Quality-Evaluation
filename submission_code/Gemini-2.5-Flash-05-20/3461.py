from typing import List

class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])

        # Initialize min/max row and column indices.
        # These are set to values outside the possible range of indices
        # to ensure that the first '1' found correctly sets the initial bounds.
        min_r = rows  # A row index cannot be >= rows
        max_r = -1    # A row index cannot be < 0
        min_c = cols  # A column index cannot be >= cols
        max_c = -1    # A column index cannot be < 0

        # Iterate through the grid to find the extreme coordinates of all '1's.
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    # Update the minimum row index found so far
                    min_r = min(min_r, r)
                    # Update the maximum row index found so far
                    max_r = max(max_r, r)
                    # Update the minimum column index found so far
                    min_c = min(min_c, c)
                    # Update the maximum column index found so far
                    max_c = max(max_c, c)

        # The problem statement guarantees that there is at least one '1' in the grid.
        # This means min_r, max_r, min_c, max_c will always be updated to valid indices.

        # Calculate the height of the bounding rectangle.
        # It's the difference between max and min row indices, plus one (to include both ends).
        height = max_r - min_r + 1

        # Calculate the width of the bounding rectangle.
        # It's the difference between max and min column indices, plus one.
        width = max_c - min_c + 1

        # The minimum area is the product of its height and width.
        return height * width