from typing import List

class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        if rows == 0:
            return 0
        cols = len(grid[0])
        
        min_row = float('inf')
        max_row = -1
        min_col = float('inf')
        max_col = -1
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    if i < min_row:
                        min_row = i
                    if i > max_row:
                        max_row = i
                    if j < min_col:
                        min_col = j
                    if j > max_col:
                        max_col = j
        
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        return height * width