from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        N = len(grid)
        # Constraints: 1 <= grid.length, so N >= 1.
        M = len(grid[0])
        # Constraints: 1 <= grid[i].length, so M >= 1.

        # Precompute number of 1s in each row and column
        # row_ones_count[r] stores the number of 1s in row r
        # col_ones_count[c] stores the number of 1s in column c
        row_ones_count = [0] * N
        col_ones_count = [0] * M

        for r in range(N):
            for c in range(M):
                if grid[r][c] == 1:
                    row_ones_count[r] += 1
                    col_ones_count[c] += 1
        
        total_triangles = 0
        # Iterate through each cell (r, c) in the grid
        for r in range(N):
            for c in range(M):
                # If grid[r][c] is 1, this cell can serve as the vertex
                # (corner of the 90-degree angle) of a right triangle.
                if grid[r][c] == 1:
                    # To form a right triangle with (r,c) as the vertex, we need:
                    # 1. Another point (r, c_other) in the same row r, where c_other != c and grid[r][c_other] == 1.
                    # 2. Another point (r_other, c) in the same column c, where r_other != r and grid[r_other][c] == 1.

                    # Number of other 1s in row r (excluding grid[r][c] itself).
                    # Since grid[r][c] == 1, row_ones_count[r] must be at least 1.
                    # So, (row_ones_count[r] - 1) will be non-negative.
                    other_ones_in_row = row_ones_count[r] - 1
                    
                    # Number of other 1s in column c (excluding grid[r][c] itself).
                    # Similarly, col_ones_count[c] must be at least 1.
                    # So, (col_ones_count[c] - 1) will be non-negative.
                    other_ones_in_col = col_ones_count[c] - 1
                    
                    # The number of ways to choose one such point from the row and one from the column
                    # is the product of the counts. Each such choice forms a unique right triangle.
                    # If either other_ones_in_row or other_ones_in_col is 0 (meaning row_ones_count[r] or 
                    # col_ones_count[c] was 1), then no such triangles can be formed with (r,c) as the vertex.
                    # The product other_ones_in_row * other_ones_in_col will correctly be 0 in such cases.
                    if other_ones_in_row > 0 and other_ones_in_col > 0:
                         total_triangles += other_ones_in_row * other_ones_in_col
          
        return total_triangles