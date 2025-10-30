class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        count = 0

        for i in range(m):
            for j in range(n):
                if grid[i][j] == 'X':
                    for x in range(i, m):
                        for y in range(j, n):
                            if self.isValid(grid, i, j, x, y):
                                count += 1

        return count

    def isValid(self, grid, i, j, x, y):
        x_count = 0
        y_count = 0
        for row in range(i, x+1):
            for col in range(j, y+1):
                if grid[row][col] == 'X':
                    x_count += 1
                elif grid[row][col] == 'Y':
                    y_count += 1
        return x_count == y_count and x_count > 0