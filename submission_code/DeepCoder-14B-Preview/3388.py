class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0
        num_rows = len(grid)
        num_cols = len(grid[0]) if num_rows > 0 else 0
        
        # Calculate the number of 1s in each row
        row_counts = [sum(row) for row in grid]
        
        # Calculate the number of 1s in each column
        col_counts = [0] * num_cols
        for j in range(num_cols):
            for i in range(num_rows):
                col_counts[j] += grid[i][j]
        
        total = 0
        for i in range(num_rows):
            for j in range(num_cols):
                if grid[i][j] == 1:
                    row_contribution = row_counts[i] - 1
                    col_contribution = col_counts[j] - 1
                    total += row_contribution * col_contribution
        
        return total