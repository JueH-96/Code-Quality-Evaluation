class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        
        # Precompute the number of 1s in each row and each column
        row_counts = [0] * rows
        col_counts = [0] * cols
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    row_counts[i] += 1
                    col_counts[j] += 1
        
        total = 0
        
        # Iterate through each cell to count the number of right triangles
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    # The number of right triangles with this cell as the right angle
                    # is (number of 1s in the same row - 1) * (number of 1s in the same column - 1)
                    total += (row_counts[i] - 1) * (col_counts[j] - 1)
        
        return total