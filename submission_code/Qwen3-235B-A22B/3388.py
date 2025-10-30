from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        
        # Calculate the number of 1s in each row
        count_row = [sum(row) for row in grid]
        
        # Calculate the number of 1s in each column
        count_col = [0] * cols
        for j in range(cols):
            total = 0
            for i in range(rows):
                total += grid[i][j]
            count_col[j] = total
        
        total_triangles = 0
        
        # Iterate through each cell to calculate possible triangles
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    total_triangles += (count_row[i] - 1) * (count_col[j] - 1)
        
        return total_triangles