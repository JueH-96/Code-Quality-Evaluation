class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        
        # Count the number of 1s in each row and column
        row_count = [0] * rows
        col_count = [0] * cols
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    row_count[r] += 1
                    col_count[c] += 1
        
        triangles = 0
        
        # Iterate through each cell in the grid
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    # For each 1 found, calculate the number of right triangles
                    # that can be formed with this point as the right angle
                    triangles += (row_count[r] - 1) * (col_count[c] - 1)
        
        return triangles