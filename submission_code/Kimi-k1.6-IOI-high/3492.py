class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]:
            return 0
        rows = len(grid)
        cols = len(grid[0])
        prefix_x = [[0] * (cols + 1) for _ in range(rows + 1)]
        prefix_y = [[0] * (cols + 1) for _ in range(rows + 1)]
        
        for i in range(rows):
            for j in range(cols):
                val_x = 1 if grid[i][j] == 'X' else 0
                val_y = 1 if grid[i][j] == 'Y' else 0
                prefix_x[i+1][j+1] = prefix_x[i][j+1] + prefix_x[i+1][j] - prefix_x[i][j] + val_x
                prefix_y[i+1][j+1] = prefix_y[i][j+1] + prefix_y[i+1][j] - prefix_y[i][j] + val_y
        
        count = 0
        for r in range(rows):
            for c in range(cols):
                sum_x = prefix_x[r+1][c+1]
                sum_y = prefix_y[r+1][c+1]
                if sum_x == sum_y and sum_x >= 1:
                    count += 1
        return count