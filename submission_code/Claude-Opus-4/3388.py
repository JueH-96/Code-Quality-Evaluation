class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        # Count 1s in each row
        row_counts = [0] * m
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    row_counts[i] += 1
        
        # Count 1s in each column
        col_counts = [0] * n
        for j in range(n):
            for i in range(m):
                if grid[i][j] == 1:
                    col_counts[j] += 1
        
        # Count right triangles
        result = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    # This cell can be the corner of a right triangle
                    # Count other 1s in the same row and column
                    row_ones = row_counts[i] - 1  # Subtract 1 to exclude current cell
                    col_ones = col_counts[j] - 1  # Subtract 1 to exclude current cell
                    result += row_ones * col_ones
        
        return result