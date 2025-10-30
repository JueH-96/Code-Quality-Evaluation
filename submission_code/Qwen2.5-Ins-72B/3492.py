from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        rows, cols = len(grid), len(grid[0])
        prefix_sum = [[[0, 0] for _ in range(cols + 1)] for _ in range(rows + 1)]
        
        # Compute prefix sums for 'X' and 'Y'
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                prefix_sum[i][j][0] = prefix_sum[i-1][j][0] + prefix_sum[i][j-1][0] - prefix_sum[i-1][j-1][0]
                prefix_sum[i][j][1] = prefix_sum[i-1][j][1] + prefix_sum[i][j-1][1] - prefix_sum[i-1][j-1][1]
                if grid[i-1][j-1] == 'X':
                    prefix_sum[i][j][0] += 1
                elif grid[i-1][j-1] == 'Y':
                    prefix_sum[i][j][1] += 1
        
        def count_equal_XY(r1, c1, r2, c2):
            x_count = prefix_sum[r2][c2][0] - prefix_sum[r1][c2][0] - prefix_sum[r2][c1][0] + prefix_sum[r1][c1][0]
            y_count = prefix_sum[r2][c2][1] - prefix_sum[r1][c2][1] - prefix_sum[r2][c1][1] + prefix_sum[r1][c1][1]
            return x_count == y_count and x_count > 0
        
        count = 0
        for r1 in range(rows):
            for c1 in range(cols):
                for r2 in range(r1 + 1, rows + 1):
                    for c2 in range(c1 + 1, cols + 1):
                        if count_equal_XY(r1, c1, r2, c2):
                            count += 1
        
        return count