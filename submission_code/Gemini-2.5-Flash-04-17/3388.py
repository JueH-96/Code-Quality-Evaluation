from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        # Get dimensions of the grid
        R = len(grid)
        C = len(grid[0])

        # Compute the number of 1s in each row and each column
        # This precomputation allows us to quickly find the number of
        # potential horizontal and vertical triangle sides originating
        # from any cell.
        row_ones = [0] * R
        col_ones = [0] * C

        # Iterate through the grid once to populate row_ones and col_ones
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    row_ones[r] += 1
                    col_ones[c] += 1

        # Initialize the total count of right triangles
        total_triangles = 0

        # Iterate through the grid again to count the triangles
        # We can count each right triangle by considering each cell (r, c)
        # that contains a '1' as a potential right-angle vertex.
        for r in range(R):
            for c in range(C):
                # A cell (r, c) can be the right-angle vertex of a right triangle
                # only if the cell itself contains a '1'.
                if grid[r][c] == 1:
                    # If grid[r][c] is the right-angle vertex, the other two points
                    # of the triangle must be:
                    # 1. Another '1' located in the same row r, but at a different column c'.
                    # 2. Another '1' located in the same column c, but at a different row r'.
                    
                    # The number of other '1's available in the same row r
                    # (excluding the current cell grid[r][c]) is row_ones[r] - 1.
                    
                    # The number of other '1's available in the same column c
                    # (excluding the current cell grid[r][c]) is col_ones[c] - 1.
                    
                    # To form a right triangle with (r, c) as the vertex, we need to choose
                    # one '1' from the available others in row r and one '1' from the
                    # available others in column c.
                    # The number of ways to make these two choices is the product of
                    # the number of options for each choice:
                    # (number of other '1's in row r) * (number of other '1's in column c).
                    # This product is (row_ones[r] - 1) * (col_ones[c] - 1).
                    
                    # If either (row_ones[r] - 1) or (col_ones[c] - 1) is 0 (meaning there are
                    # less than two '1's in that row or column including the vertex),
                    # the product will be 0, correctly indicating that no right triangle
                    # can be formed with (r, c) as the right-angle vertex in that case.
                    
                    total_triangles += (row_ones[r] - 1) * (col_ones[c] - 1)

        return total_triangles