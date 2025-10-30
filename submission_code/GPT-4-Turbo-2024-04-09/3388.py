class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        
        # Count of 1s in each row and each column
        row_count = [0] * rows
        col_count = [0] * cols
        
        # Fill row_count and col_count
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
                    # (row_count[r] - 1) choices in the same row
                    # (col_count[c] - 1) choices in the same column
                    right_triangles += (row_count[r] - 1) * (col_count[c] - 1)
        
        return right_triangles