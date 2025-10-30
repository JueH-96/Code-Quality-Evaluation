from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        """
        Calculates the number of right triangles formed by '1's in a grid.
        
        A right triangle's vertex is a '1' that has at least one other '1' in its
        row and at least one other '1' in its column. This method pre-computes
        the number of '1's in each row and column to efficiently calculate the result.
        """
        # Constraints: 1 <= grid.length, grid[i].length <= 1000.
        # The grid is non-empty, and rows are non-empty.
        rows = len(grid)
        cols = len(grid[0])
        
        # Step 1: Pre-compute the number of '1's in each row and column.
        row_counts = [0] * rows
        col_counts = [0] * cols
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    row_counts[r] += 1
                    col_counts[c] += 1
                    
        # Step 2: Calculate the number of triangles.
        # A right triangle is formed by three '1's at (r, c), (r, c'), and (r', c).
        # We iterate through each '1' and treat it as the vertex of the right angle.
        total_triangles = 0
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    # Number of other '1's in the same row to form the horizontal leg.
                    # Since grid[r][c] is 1, row_counts[r] is at least 1.
                    others_in_row = row_counts[r] - 1
                    
                    # Number of other '1's in the same column to form the vertical leg.
                    # Since grid[r][c] is 1, col_counts[c] is at least 1.
                    others_in_col = col_counts[c] - 1
                    
                    # The number of triangles with (r, c) as the vertex is the
                    # product of choices for the other two points. The product
                    # will be non-negative.
                    total_triangles += others_in_row * others_in_col
                        
        return total_triangles