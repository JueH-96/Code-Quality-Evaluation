import math
from typing import List # Ensure List is imported if running locally or required by environment

class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        """
        Calculates the minimum area of a rectangle that encloses all the 1s in a 2D binary grid.

        The approach is to find the minimum and maximum row and column indices that contain a '1'.
        These four values define the boundaries of the smallest rectangle enclosing all '1's.

        Args:
            grid: A list of lists of integers (0 or 1) representing the 2D grid. 
                  It's guaranteed that there is at least one 1 in the grid based on constraints.

        Returns:
            An integer representing the minimum possible area of the enclosing rectangle.
        """
        
        rows = len(grid)
        # Check if grid is empty, although constraints state rows >= 1
        if rows == 0:
            return 0 
            
        cols = len(grid[0])
        # Check if rows are empty, although constraints state cols >= 1
        if cols == 0:
            return 0

        # Initialize the boundaries.
        # We are looking for the smallest row index (min_r), largest row index (max_r),
        # smallest column index (min_c), and largest column index (max_c) that contain a 1.
        # Initialize min values to be larger than any possible index, and max values smaller.
        min_r = rows  # Initialize min_r to rows (one greater than the max possible row index rows-1)
        max_r = -1    # Initialize max_r to -1 (one less than the min possible row index 0)
        min_c = cols  # Initialize min_c to cols (one greater than the max possible col index cols-1)
        max_c = -1    # Initialize max_c to -1 (one less than the min possible col index 0)

        # Iterate through each cell of the grid.
        for r in range(rows):
            for c in range(cols):
                # If the cell contains a 1, update the boundaries if the current cell's index
                # extends the current boundary.
                if grid[r][c] == 1:
                    min_r = min(min_r, r) # Find the minimum row index containing a 1
                    max_r = max(max_r, r) # Find the maximum row index containing a 1
                    min_c = min(min_c, c) # Find the minimum column index containing a 1
                    max_c = max(max_c, c) # Find the maximum column index containing a 1

        # After iterating through the grid, min_r, max_r, min_c, max_c will hold the
        # coordinates defining the smallest bounding box containing all 1s.
        # The problem guarantees at least one '1' exists, so these variables will hold valid indices.

        # Calculate the height of the rectangle.
        # The height is the difference between the maximum and minimum row indices, plus one
        # because the range is inclusive. For example, if min_r=2 and max_r=4, the rows are 2, 3, 4,
        # which is a height of 3 (4 - 2 + 1).
        height = max_r - min_r + 1

        # Calculate the width of the rectangle.
        # The width is the difference between the maximum and minimum column indices, plus one
        # for inclusivity.
        width = max_c - min_c + 1

        # The minimum area is the product of the height and width.
        area = height * width

        return area