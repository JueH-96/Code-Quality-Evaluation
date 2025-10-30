class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        
        # Initialize min and max row and column indices
        min_row, max_row = rows, -1
        min_col, max_col = cols, -1
        
        # Iterate through the grid to find the boundaries
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
        
        # Calculate the area
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        return height * width