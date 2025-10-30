from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        # Compute the number of 1's in each row
        row_counts = [sum(row) for row in grid]
        
        # Compute the number of 1's in each column
        cols = len(grid[0]) if grid else 0
        col_counts = [0] * cols
        for row in grid:
            for j in range(cols):
                col_counts[j] += row[j]
        
        total = 0
        
        # Iterate through each cell to calculate contributions
        for i in range(len(grid)):
            for j in range(cols):
                if grid[i][j] == 1:
                    total += (row_counts[i] - 1) * (col_counts[j] - 1)
        
        return total