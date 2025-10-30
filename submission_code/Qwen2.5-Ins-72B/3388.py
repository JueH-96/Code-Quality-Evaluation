from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        row_counts = [0] * rows
        col_counts = [0] * cols
        
        # Count the number of 1s in each row and column
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    row_counts[i] += 1
                    col_counts[j] += 1
        
        count = 0
        # Check each cell that is 1
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    # Number of right triangles with this cell as the right angle
                    count += (row_counts[i] - 1) * (col_counts[j] - 1)
        
        return count