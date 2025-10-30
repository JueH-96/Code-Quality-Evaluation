from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        # Get dimensions of the grid
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        
        # Precompute row counts of 1's
        row_counts = [0] * rows
        for i in range(rows):
            row_counts[i] = sum(grid[i])
        
        # Precompute column counts of 1's
        col_counts = [0] * cols
        for j in range(cols):
            col_counts[j] = sum(grid[i][j] for i in range(rows))
        
        # Calculate the total number of right triangles
        total = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    # Number of 1's in row i excluding grid[i][j]
                    a = row_counts[i] - 1
                    # Number of 1's in column j excluding grid[i][j]
                    b = col_counts[j] - 1
                    # Add the product to the total
                    total += a * b
        return total