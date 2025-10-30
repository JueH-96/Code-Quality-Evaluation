class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        
        # Count the number of 1s in each row and each column
        row_count = [0] * rows
        col_count = [0] * cols
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    row_count[r] += 1
                    col_count[c] += 1
        
        # Calculate the number of right triangles
        right_triangles = 0
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    # The number of right triangles with (r, c) as the right angle
                    right_triangles += (row_count[r] - 1) * (col_count[c] - 1)
        
        return right_triangles