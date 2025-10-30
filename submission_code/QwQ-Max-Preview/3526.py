class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        sum_row = 0
        for row in grid:
            n = len(row)
            count = 0
            for i in range(n // 2):
                if row[i] != row[n - 1 - i]:
                    count += 1
            sum_row += count
        
        sum_col = 0
        m = len(grid)
        n_cols = len(grid[0]) if m else 0
        for j in range(n_cols):
            column = [grid[i][j] for i in range(m)]
            len_col = len(column)
            count = 0
            for i in range(len_col // 2):
                if column[i] != column[len_col - 1 - i]:
                    count += 1
            sum_col += count
        
        return min(sum_row, sum_col)