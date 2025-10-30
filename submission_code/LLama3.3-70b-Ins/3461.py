from typing import List

class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        # Find the minimum and maximum row and column indices that contain 1
        min_row, max_row = float('inf'), float('-inf')
        min_col, max_col = float('inf'), float('-inf')
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    min_row, max_row = min(min_row, i), max(max_row, i)
                    min_col, max_col = min(min_col, j), max(max_col, j)
        
        # Calculate the area of the rectangle
        area = (max_row - min_row + 1) * (max_col - min_col + 1)
        
        return area