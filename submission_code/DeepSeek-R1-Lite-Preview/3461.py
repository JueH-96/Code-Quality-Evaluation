class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        # Initialize min and max row and column indices
        min_row = float('inf')
        max_row = -float('inf')
        min_col = float('inf')
        max_col = -float('inf')
        
        # Iterate through each cell in the grid
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    # Update min and max row and column indices
                    min_row = min(min_row, i)
                    max_row = max(max_row, i)
                    min_col = min(min_col, j)
                    max_col = max(max_col, j)
        
        # Calculate the area of the rectangle
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        area = height * width
        return area