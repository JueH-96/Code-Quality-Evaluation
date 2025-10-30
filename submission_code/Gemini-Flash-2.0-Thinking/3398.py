class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        def check_square(current_grid):
            for r in range(2):
                for c in range(2):
                    color = current_grid[r][c]
                    if (current_grid[r + 0][c + 0] == color and
                            current_grid[r + 0][c + 1] == color and
                            current_grid[r + 1][c + 0] == color and
                            current_grid[r + 1][c + 1] == color):
                        return True
            return False

        # Check if there is already a 2x2 square of the same color
        if check_square(grid):
            return True

        # Try changing one cell at a time
        for r in range(3):
            for c in range(3):
                original_color = grid[r][c]
                opposite_color = 'W' if original_color == 'B' else 'B'

                # Change the color of the current cell
                grid[r] = list(grid[r])
                grid[r][c] = opposite_color
                grid[r] = "".join(grid[r])

                # Check if a 2x2 square of the same color exists after the change
                if check_square(grid):
                    return True

                # Revert the change
                grid[r] = list(grid[r])
                grid[r][c] = original_color
                grid[r] = "".join(grid[r])

        return False