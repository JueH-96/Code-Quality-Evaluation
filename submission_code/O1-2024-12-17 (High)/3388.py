class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        
        # Count the number of 1s in each row
        row_count = [sum(row) for row in grid]
        
        # Count the number of 1s in each column
        col_count = [0]*C
        for r in range(R):
            for c in range(C):
                col_count[c] += grid[r][c]
        
        # Calculate the number of right triangles
        # For each cell (r, c) that is 1, the number of right triangles
        # with (r, c) as the "corner" is:
        # (row_count[r] - 1) * (col_count[c] - 1)
        result = 0
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    result += (row_count[r] - 1) * (col_count[c] - 1)
        
        return result