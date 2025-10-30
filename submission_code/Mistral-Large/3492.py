from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]:
            return 0

        rows, cols = len(grid), len(grid[0])
        count = 0

        # Precompute the prefix sums for 'X' and 'Y'
        prefix_x = [[0] * (cols + 1) for _ in range(rows + 1)]
        prefix_y = [[0] * (cols + 1) for _ in range(rows + 1)]

        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                prefix_x[i][j] = prefix_x[i-1][j] + prefix_x[i][j-1] - prefix_x[i-1][j-1] + (1 if grid[i-1][j-1] == 'X' else 0)
                prefix_y[i][j] = prefix_y[i-1][j] + prefix_y[i][j-1] - prefix_y[i-1][j-1] + (1 if grid[i-1][j-1] == 'Y' else 0)

        # Check all submatrices
        for r1 in range(1, rows + 1):
            for c1 in range(1, cols + 1):
                for r2 in range(r1, rows + 1):
                    for c2 in range(c1, cols + 1):
                        x_count = prefix_x[r2][c2] - prefix_x[r1-1][c2] - prefix_x[r2][c1-1] + prefix_x[r1-1][c1-1]
                        y_count = prefix_y[r2][c2] - prefix_y[r1-1][c2] - prefix_y[r2][c1-1] + prefix_y[r1-1][c1-1]
                        if x_count > 0 and x_count == y_count:
                            count += 1

        return count