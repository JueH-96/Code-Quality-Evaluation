class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        count = 0
        for r1 in range(m):
            for c1 in range(n):
                if grid[r1][c1] == '.':
                    continue
                for r2 in range(r1, m):
                    for c2 in range(c1, n):
                        sub_grid = [row[c1:c2+1] for row in grid[r1:r2+1]]
                        x_count = 0
                        y_count = 0
                        x_present = False
                        for row in sub_grid:
                            for cell in row:
                                if cell == 'X':
                                    x_count += 1
                                    x_present = True
                                elif cell == 'Y':
                                    y_count += 1
                        if x_present and x_count == y_count:
                            count += 1
        return count