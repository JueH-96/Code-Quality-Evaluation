from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        # Number of rows
        n = len(grid)
        # Number of columns
        m = len(grid[0])
        
        # Count of 1s in each row
        row_counts = [0] * n
        # Count of 1s in each column
        col_counts = [0] * m
        
        # Populate row_counts and col_counts
        for r in range(n):
            for c in range(m):
                if grid[r][c] == 1:
                    row_counts[r] += 1
                    col_counts[c] += 1
        
        # Calculate the number of right triangles
        result = 0
        for r in range(n):
            for c in range(m):
                if grid[r][c] == 1:
                    # (row_counts[r] - 1) is how many other 1's in the same row
                    # (col_counts[c] - 1) is how many other 1's in the same column
                    # Their product gives the number of distinct right triangles
                    # with (r, c) as the vertex (corner).
                    result += (row_counts[r] - 1) * (col_counts[c] - 1)
        
        return result