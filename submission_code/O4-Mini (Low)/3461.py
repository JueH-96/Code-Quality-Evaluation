from typing import List

class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        # Initialize boundaries
        min_row, max_row = float('inf'), -float('inf')
        min_col, max_col = float('inf'), -float('inf')
        
        # Iterate over the grid to find the extents of '1's
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                if val == 1:
                    if i < min_row: min_row = i
                    if i > max_row: max_row = i
                    if j < min_col: min_col = j
                    if j > max_col: max_col = j
        
        # Compute height and width of the bounding rectangle
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        
        return height * width