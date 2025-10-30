class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        def count_submatrices_with_equal_XY_and_at_least_one_X(grid):
            rows, cols = len(grid), len(grid[0])
            count = 0

            for r1 in range(rows):
                for c1 in range(cols):
                    if grid[r1][c1] == 'X':
                        for r2 in range(r1, rows):
                            for c2 in range(c1, cols):
                                if self.is_valid_submatrix(grid, r1, c1, r2, c2):
                                    count += 1
            return count

        def is_valid_submatrix(self, grid, r1, c1, r2, c2):
            count_X = 0
            count_Y = 0
            for i in range(r1, r2 + 1):
                for j in range(c1, c2 + 1):
                    if grid[i][j] == 'X':
                        count_X += 1
                    elif grid[i][j] == 'Y':
                        count_Y += 1
            return count_X > 0 and count_X == count_Y

        return count_submatrices_with_equal_XY_and_at_least_one_X(grid)