class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        # Initialize the bounds for the rectangle
        min_row, max_row = float('inf'), float('-inf')
        min_col, max_col = float('inf'), float('-inf')
        
        # Iterate over the grid to find the bounds of the rectangle
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 1:
                    min_row = min(min_row, i)
                    max_row = max(max_row, i)
                    min_col = min(min_col, j)
                    max_col = max(max_col, j)
        
        # Calculate the area of the rectangle
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        return height * width