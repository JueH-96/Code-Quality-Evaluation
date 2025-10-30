from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        row_count = [0] * rows
        col_count = [0] * cols
        total_triangles = 0
        
        # Count the number of 1s in each row and column
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    row_count[i] += 1
                    col_count[j] += 1
        
        # Calculate the number of right triangles
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    # Number of right triangles with grid[i][j] as the right angle
                    total_triangles += (row_count[i] - 1) * (col_count[j] - 1)
        
        return total_triangles