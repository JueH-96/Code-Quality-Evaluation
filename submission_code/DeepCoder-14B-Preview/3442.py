class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0
        
        min_row = len(grid)  # Initialize to a value larger than any possible row index
        max_row = -1
        min_col = len(grid[0])  # Initialize to a value larger than any possible column index
        max_col = -1
        
        for i in range(len(grid)):
            for j in range(len(grid[i])):
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