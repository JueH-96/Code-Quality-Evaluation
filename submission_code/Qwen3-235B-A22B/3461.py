class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        min_row = float('inf')
        max_row = -float('inf')
        min_col = float('inf')
        max_col = -float('inf')
        
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        
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