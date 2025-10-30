from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        count = 0
        
        # Precompute the number of 'X' and 'Y' up to each cell
        prefix_sum = [[[0, 0] for _ in range(cols + 1)] for _ in range(rows + 1)]
        
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                prefix_sum[i][j][0] = prefix_sum[i-1][j][0] + prefix_sum[i][j-1][0] - prefix_sum[i-1][j-1][0] + (1 if grid[i-1][j-1] == 'X' else 0)
                prefix_sum[i][j][1] = prefix_sum[i-1][j][1] + prefix_sum[i][j-1][1] - prefix_sum[i-1][j-1][1] + (1 if grid[i-1][j-1] == 'Y' else 0)
        
        # Check all submatrices starting from (0, 0)
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                x_count = prefix_sum[i][j][0]
                y_count = prefix_sum[i][j][1]
                
                if x_count > 0 and x_count == y_count:
                    count += 1
        
        return count