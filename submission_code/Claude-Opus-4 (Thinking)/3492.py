class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        count = 0
        
        # For each column, track cumulative X and Y counts up to current row
        col_counts = [[0, 0] for _ in range(n)]  # [x_count, y_count] for each column
        
        for i in range(m):
            row_x = 0  # X count in submatrix from (0,0) to (i,j)
            row_y = 0  # Y count in submatrix from (0,0) to (i,j)
            
            for j in range(n):
                # Update column counts for current row
                if grid[i][j] == 'X':
                    col_counts[j][0] += 1
                elif grid[i][j] == 'Y':
                    col_counts[j][1] += 1
                
                # Add this column's counts to running total
                row_x += col_counts[j][0]
                row_y += col_counts[j][1]
                
                # Check if submatrix from (0,0) to (i,j) is valid
                if row_x == row_y and row_x > 0:
                    count += 1
        
        return count