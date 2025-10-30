class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        min_row, max_row = float('inf'), float('-inf')
        min_col, max_col = float('inf'), float('-inf')
        
        # Find the boundaries of all 1's in the grid
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    min_row = min(min_row, i)
                    max_row = max(max_row, i)
                    min_col = min(min_col, j)
                    max_col = max(max_col, j)
        
        # Calculate the height and width of the rectangle
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        
        # Return the area
        return height * width