class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0  # Assuming grid has at least one 1 as per constraints
        
        num_rows = len(grid)
        num_cols = len(grid[0])
        
        # Initialize min and max row and column indices
        min_row = num_rows - 1
        max_row = 0
        min_col = num_cols - 1
        max_col = 0
        
        # Find the bounds of rows and columns that contain at least one 1
        for i in range(num_rows):
            for j in range(num_cols):
                if grid[i][j] == 1:
                    if i < min_row:
                        min_row = i
                    if i > max_row:
                        max_row = i
                    if j < min_col:
                        min_col = j
                    if j > max_col:
                        max_col = j
        
        # Calculate the area of the rectangle
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        area = height * width
        return area