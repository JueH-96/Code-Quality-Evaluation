class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        def has_square(current_grid):
            for r in range(2):
                for c in range(2):
                    color = current_grid[r][c]
                    if (current_grid[r+1][c] == color and
                        current_grid[r][c+1] == color and
                        current_grid[r+1][c+1] == color):
                        return True
            return False

        if has_square(grid):
            return True

        for r in range(3):
            for c in range(3):
                original_color = grid[r][c]
                new_grid = [row[:] for row in grid] # Create a copy
                new_grid[r][c] = 'W' if original_color == 'B' else 'B'
                if has_square(new_grid):
                    return True

        return False