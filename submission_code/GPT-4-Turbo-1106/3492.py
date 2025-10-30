class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        def count_submatrices(r1, r2, c1, c2):
            x_count = y_count = 0
            for i in range(r1, r2 + 1):
                for j in range(c1, c2 + 1):
                    if grid[i][j] == 'X':
                        x_count += 1
                    elif grid[i][j] == 'Y':
                        y_count += 1
            return x_count == y_count and x_count > 0

        rows = len(grid)
        cols = len(grid[0])
        count = 0

        # Iterate over all possible submatrices that include grid[0][0]
        for r2 in range(rows):
            for c2 in range(cols):
                if count_submatrices(0, r2, 0, c2):
                    count += 1

        return count