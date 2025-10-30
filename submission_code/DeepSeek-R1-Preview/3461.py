class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        min_row = float('inf')
        max_row = -float('inf')
        min_col = float('inf')
        max_col = -float('inf')
        
        rows = len(grid)
        if rows == 0:
            return 0  # Though per constraints, this won't happen as grid has at least one 1
        cols = len(grid[0])
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    min_row = min(min_row, i)
                    max_row = max(max_row, i)
                    min_col = min(min_col, j)
                    max_col = max(max_col, j)
        
        # Calculate the height and width
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        
        return height * width