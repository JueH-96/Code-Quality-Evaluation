import collections # Note: collections is not actually used, but included if needed elsewhere.
from typing import List # Required for type hinting List[List[int]]

class Solution:
    """
    This class provides a solution to count the number of right triangles
    formed by cells with value 1 in a 2D boolean matrix.
    """
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        """
        Counts the number of right triangles formed by three '1's in the grid.

        A right triangle is formed by three points (r1, c1), (r1, c2), and (r2, c1)
        such that grid[r1][c1]=1, grid[r1][c2]=1, and grid[r2][c1]=1,
        and r1 != r2, c1 != c2. The point (r1, c1) is the vertex where the 
        right angle is formed.

        The algorithm works by first precomputing the number of '1's in each row
        and each column. Then, it iterates through each cell of the grid. If a cell
        (r, c) contains a '1', it can potentially be the vertex of a right angle
        of a right triangle. The number of other '1's in the same row `r` (excluding
        the vertex itself) gives the number of possible horizontal points. The number
        of other '1's in the same column `c` (excluding the vertex itself) gives the
        number of possible vertical points. The product of these two counts gives the
        number of right triangles that have (r, c) as their right-angle vertex.
        Summing these products over all cells containing '1' gives the total number
        of right triangles.

        Args:
          grid: A 2D list of integers, where grid[i][j] is either 0 or 1.

        Returns:
          An integer representing the total number of right triangles formed by
          three '1's in the grid according to the definition.
        """
        rows = len(grid)
        # Per constraints, rows >= 1, but check for robustness
        if rows == 0:
            return 0
        
        cols = len(grid[0])
        # Per constraints, cols >= 1, but check for robustness
        if cols == 0:
            return 0

        # Precompute the number of 1s in each row and column
        # Initialize lists to store counts for rows and columns
        ones_in_row = [0] * rows
        ones_in_col = [0] * cols

        # First pass: iterate through the grid to populate the counts
        # Time complexity: O(rows * cols)
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    ones_in_row[r] += 1
                    ones_in_col[c] += 1

        # Initialize the total count of right triangles
        triangle_count = 0

        # Second pass: iterate through the grid again
        # For each cell containing '1', consider it as the potential vertex
        # of the right angle of a triangle.
        # Time complexity: O(rows * cols)
        for r in range(rows):
            for c in range(cols):
                # If the current cell (r, c) is 1
                if grid[r][c] == 1:
                    # Calculate the number of other '1's in the same row `r`
                    # These points can form the horizontal leg of the triangle.
                    # We subtract 1 because we don't count the vertex (r, c) itself.
                    row_others = ones_in_row[r] - 1
                    
                    # Calculate the number of other '1's in the same column `c`
                    # These points can form the vertical leg of the triangle.
                    # We subtract 1 because we don't count the vertex (r, c) itself.
                    col_others = ones_in_col[c] - 1

                    # To form a right triangle with (r, c) as the vertex,
                    # we need at least one other '1' in the same row (row_others > 0)
                    # and at least one other '1' in the same column (col_others > 0).
                    if row_others > 0 and col_others > 0:
                        # The number of triangles with (r, c) as the vertex is
                        # the product of the number of choices for the other two points.
                        # It's (number of choices for horizontal point) * (number of choices for vertical point).
                        triangle_count += row_others * col_others

        # Return the total count accumulated
        # The total time complexity is O(rows * cols) due to two passes over the grid.
        # The space complexity is O(rows + cols) for storing the counts.
        return triangle_count