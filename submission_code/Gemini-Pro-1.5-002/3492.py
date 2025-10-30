class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        count = 0

        for r1 in range(rows):
            for c1 in range(cols):
                x_count = 0
                y_count = 0
                has_x = False
                for r2 in range(r1, rows):
                    for c2 in range(c1, cols):
                        if grid[r2][c2] == 'X':
                            x_count += 1
                            has_x = True
                        elif grid[r2][c2] == 'Y':
                            y_count += 1
                        
                        if has_x and x_count == y_count:
                            count += 1
        return count