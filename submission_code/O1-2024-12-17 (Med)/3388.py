class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        # Count of rows and columns
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        
        row_counts = [0]*n
        col_counts = [0]*m
        
        # Compute how many 1s in each row and column
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    row_counts[i] += 1
                    col_counts[j] += 1
        
        # Calculate number of right triangles
        result = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    # (row_counts[i] - 1) ways to pick another point in this row
                    # (col_counts[j] - 1) ways to pick another point in this column
                    result += (row_counts[i] - 1) * (col_counts[j] - 1)
        
        return result