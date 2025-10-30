class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        
        # Find min/max coordinates of 1's
        min_row = m
        max_row = -1
        min_col = n 
        max_col = -1
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    min_row = min(min_row, i)
                    max_row = max(max_row, i)
                    min_col = min(min_col, j)
                    max_col = max(max_col, j)
        
        # Calculate area
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        
        return height * width