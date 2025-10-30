from typing import List

class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        # Initialize boundaries with extreme opposite values
        min_row, max_row = float('inf'), -float('inf')
        min_col, max_col = float('inf'), -float('inf')
        
        # Iterate through the grid to find the extreme 1-positions
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                if val == 1:
                    min_row = min(min_row, r)
                    max_row = max(max_row, r)
                    min_col = min(min_col, c)
                    max_col = max(max_col, c)
        
        # Height and width of the smallest rectangle that encloses all 1's
        height = max_row - min_row + 1
        width  = max_col - min_col + 1
        
        return height * width