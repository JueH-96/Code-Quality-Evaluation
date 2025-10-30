class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        count = 0

        for r1 in range(rows):
            for c1 in range(cols):
                if r1 == 0 and c1 == 0:
                    for r2 in range(r1, rows):
                        for c2 in range(c1, cols):
                            x_count = 0
                            y_count = 0
                            has_x = False
                            for i in range(r1, r2 + 1):
                                for j in range(c1, c2 + 1):
                                    if grid[i][j] == 'X':
                                        x_count += 1
                                        has_x = True
                                    elif grid[i][j] == 'Y':
                                        y_count += 1
                            if has_x and x_count == y_count:
                                count += 1
        return count