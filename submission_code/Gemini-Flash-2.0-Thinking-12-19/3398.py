class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        def check_square(current_grid):
            for r in range(2):
                for c in range(2):
                    color = current_grid[r][c]
                    if (current_grid[r+1][c] == color and
                        current_grid[r][c+1] == color and
                        current_grid[r+1][c+1] == color):
                        return True
            return False

        if check_square(grid):
            return True

        for r in range(3):
            for c in range(3):
                original_color = grid[r][c]
                opposite_color = 'W' if original_color == 'B' else 'B'
                grid[r][c] = opposite_color
                if check_square(grid):
                    return True
                grid[r][c] = original_color # revert back

        return False