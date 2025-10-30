class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        count = 0
        for r in range(rows):
            for c in range(cols):
                x_count = 0
                y_count = 0
                for i in range(r + 1):
                    for j in range(c + 1):
                        if grid[i][j] == 'X':
                            x_count += 1
                        elif grid[i][j] == 'Y':
                            y_count += 1
                if x_count == y_count and x_count > 0:
                    count += 1
        return count