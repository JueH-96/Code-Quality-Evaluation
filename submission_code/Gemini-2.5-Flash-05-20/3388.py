import collections
from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        R = len(grid)
        C = len(grid[0])

        # Precompute counts of '1's in each row and column.
        # This is done in a single pass over the grid, taking O(R*C) time.
        row_ones = [0] * R
        col_ones = [0] * C
        
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    row_ones[r] += 1
                    col_ones[c] += 1
        
        # Initialize total count of right triangles.
        total_triangles = 0
        
        # Iterate through each cell to find potential right-angle vertices.
        # This is another pass over the grid, also taking O(R*C) time.
        for r in range(R):
            for c in range(C):
                # If the current cell grid[r][c] is '1', it can be the vertex of a right triangle.
                # A right triangle needs three '1's: (r, c), (r, c'), and (r', c).
                # (r, c) is the vertex at the right angle.
                if grid[r][c] == 1:
                    # 'row_ones[r]' is the total number of '1's in row 'r'.
                    # 'row_ones[r] - 1' gives the number of *other* '1's in the same row,
                    # which can serve as the horizontal leg of the triangle.
                    num_horizontal_leg_points = row_ones[r] - 1
                    
                    # 'col_ones[c]' is the total number of '1's in column 'c'.
                    # 'col_ones[c] - 1' gives the number of *other* '1's in the same column,
                    # which can serve as the vertical leg of the triangle.
                    num_vertical_leg_points = col_ones[c] - 1
                    
                    # For each chosen horizontal leg point and each chosen vertical leg point,
                    # a unique right triangle is formed with (r, c) as the vertex.
                    # The product accounts for all such combinations.
                    # If either `num_horizontal_leg_points` or `num_vertical_leg_points` is 0,
                    # their product will be 0, correctly adding no triangles.
                    total_triangles += num_horizontal_leg_points * num_vertical_leg_points
                        
        return total_triangles